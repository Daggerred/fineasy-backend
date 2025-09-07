"""
GST Compliance Service - India-specific compliance checking
"""
from typing import List, Dict, Any, Optional
import logging
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from ..models.base import ComplianceIssue, ComplianceType, ComplianceSeverity, ComplianceStatus
from ..models.responses import ComplianceResponse, GSTValidationResult, TaxVerificationResult
from ..database import DatabaseManager
from ..config import settings

logger = logging.getLogger(__name__)


class ComplianceChecker:
    """GST Compliance checking service for Indian businesses"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.gst_api_key = settings.GST_API_KEY
        self.gst_api_url = settings.GST_API_URL
        
        # GST rate configurations
        self.gst_rates = {
            'exempt': 0.0,
            'zero_rated': 0.0,
            'standard': 18.0,
            'reduced_1': 5.0,
            'reduced_2': 12.0,
            'luxury': 28.0
        }
        
        # Required fields for GST compliance
        self.required_invoice_fields = [
            'invoice_number', 'invoice_date', 'supplier_gstin', 'customer_gstin',
            'place_of_supply', 'items', 'taxable_value', 'tax_amount', 'total_amount'
        ]
    
    async def check_compliance(self, invoice_id: str) -> ComplianceResponse:
        """Check comprehensive invoice compliance"""
        logger.info(f"Checking compliance for invoice: {invoice_id}")
        
        try:
            # Get invoice data from database
            invoice_data = await self._get_invoice_data(invoice_id)
            if not invoice_data:
                return ComplianceResponse(
                    success=False,
                    message="Invoice not found",
                    invoice_id=invoice_id,
                    issues=[],
                    overall_status=ComplianceStatus.CRITICAL_ISSUES,
                    compliance_score=0.0
                )
            
            # Run all compliance checks
            issues = []
            
            # 1. Field completeness validation
            field_issues = await self._check_field_completeness(invoice_data)
            issues.extend(field_issues)
            
            # 2. GST number validation
            gst_issues = await self._validate_gst_numbers(invoice_data)
            issues.extend(gst_issues)
            
            # 3. Tax calculation verification
            tax_issues = await self._verify_tax_calculations(invoice_data)
            issues.extend(tax_issues)
            
            # 4. Place of supply validation
            pos_issues = await self._validate_place_of_supply(invoice_data)
            issues.extend(pos_issues)
            
            # Calculate overall compliance score and status
            compliance_score = self._calculate_compliance_score(issues)
            overall_status = self._determine_compliance_status(issues)
            
            return ComplianceResponse(
                invoice_id=invoice_id,
                issues=issues,
                overall_status=overall_status,
                compliance_score=compliance_score
            )
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return ComplianceResponse(
                success=False,
                message=f"Compliance check failed: {str(e)}",
                invoice_id=invoice_id,
                issues=[],
                overall_status=ComplianceStatus.CRITICAL_ISSUES,
                compliance_score=0.0
            )
    
    async def validate_gst_number(self, gstin: str) -> GSTValidationResult:
        """Validate GST number using format validation and API verification"""
        logger.info(f"Validating GST number: {gstin}")
        
        try:
            # Basic format validation
            if not self._validate_gstin_format(gstin):
                return GSTValidationResult(
                    gstin=gstin,
                    is_valid=False,
                    status="invalid_format",
                    errors=["Invalid GSTIN format. Must be 15 characters: 2 digits (state) + 10 characters (PAN) + 1 digit (entity) + 1 character (Z) + 1 check digit"]
                )
            
            # API validation (if API key is available)
            if self.gst_api_key:
                api_result = await self._validate_gstin_via_api(gstin)
                return api_result
            else:
                # Fallback to format validation only
                logger.warning("GST API key not configured, using format validation only")
                return GSTValidationResult(
                    gstin=gstin,
                    is_valid=True,
                    status="format_valid",
                    errors=["API validation not available - format validation passed"]
                )
                
        except Exception as e:
            logger.error(f"Error validating GST number: {e}")
            return GSTValidationResult(
                gstin=gstin,
                is_valid=False,
                status="validation_error",
                errors=[f"Validation failed: {str(e)}"]
            )
    
    async def check_invoice_compliance(self, invoice_id: str) -> List[ComplianceIssue]:
        """Check invoice compliance and return list of issues"""
        compliance_response = await self.check_compliance(invoice_id)
        return compliance_response.issues
    
    async def verify_tax_calculations(self, invoice_id: str) -> TaxVerificationResult:
        """Verify tax calculations for an invoice"""
        logger.info(f"Verifying tax calculations for invoice: {invoice_id}")
        
        try:
            invoice_data = await self._get_invoice_data(invoice_id)
            if not invoice_data:
                return TaxVerificationResult(
                    invoice_id=invoice_id,
                    calculated_tax=0.0,
                    expected_tax=0.0,
                    variance=0.0,
                    is_correct=False,
                    breakdown={"error": "Invoice not found"}
                )
            
            # Calculate expected tax based on items and rates
            expected_tax, breakdown = await self._calculate_expected_tax(invoice_data)
            
            # Get actual tax from invoice
            actual_tax = float(invoice_data.get('tax_amount', 0))
            
            # Calculate variance
            variance = abs(expected_tax - actual_tax)
            is_correct = variance < 0.01  # Allow 1 paisa tolerance
            
            return TaxVerificationResult(
                invoice_id=invoice_id,
                calculated_tax=actual_tax,
                expected_tax=expected_tax,
                variance=variance,
                is_correct=is_correct,
                breakdown=breakdown
            )
            
        except Exception as e:
            logger.error(f"Error verifying tax calculations: {e}")
            return TaxVerificationResult(
                invoice_id=invoice_id,
                calculated_tax=0.0,
                expected_tax=0.0,
                variance=0.0,
                is_correct=False,
                breakdown={"error": str(e)}
            )
    
    # Private helper methods
    
    async def _get_invoice_data(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Get invoice data from database"""
        try:
            query = """
            SELECT i.*, b.gstin as supplier_gstin, c.gstin as customer_gstin,
                   b.state as supplier_state, c.state as customer_state
            FROM invoices i
            LEFT JOIN businesses b ON i.business_id = b.id
            LEFT JOIN customers c ON i.customer_id = c.id
            WHERE i.id = %s
            """
            result = await self.db.fetch_one(query, (invoice_id,))
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error fetching invoice data: {e}")
            return None
    
    async def _check_field_completeness(self, invoice_data: Dict[str, Any]) -> List[ComplianceIssue]:
        """Check if all required fields are present and valid"""
        issues = []
        
        for field in self.required_invoice_fields:
            if field not in invoice_data or not invoice_data[field]:
                issues.append(ComplianceIssue(
                    type=ComplianceType.MISSING_FIELDS,
                    description=f"Missing required field: {field}",
                    plain_language_explanation=f"This invoice is missing the {field.replace('_', ' ')} field, which is required for GST compliance.",
                    suggested_fixes=[f"Add the {field.replace('_', ' ')} to the invoice"],
                    severity=ComplianceSeverity.HIGH,
                    field_name=field
                ))
        
        # Check invoice number format
        if invoice_data.get('invoice_number'):
            if not re.match(r'^[A-Z0-9/-]+$', invoice_data['invoice_number']):
                issues.append(ComplianceIssue(
                    type=ComplianceType.MISSING_FIELDS,
                    description="Invalid invoice number format",
                    plain_language_explanation="Invoice number should contain only letters, numbers, hyphens, and forward slashes.",
                    suggested_fixes=["Update invoice number to use only allowed characters"],
                    severity=ComplianceSeverity.MEDIUM,
                    field_name="invoice_number",
                    current_value=invoice_data['invoice_number']
                ))
        
        return issues
    
    async def _validate_gst_numbers(self, invoice_data: Dict[str, Any]) -> List[ComplianceIssue]:
        """Validate GST numbers in the invoice"""
        issues = []
        
        # Validate supplier GSTIN
        supplier_gstin = invoice_data.get('supplier_gstin')
        if supplier_gstin:
            validation_result = await self.validate_gst_number(supplier_gstin)
            if not validation_result.is_valid:
                issues.append(ComplianceIssue(
                    type=ComplianceType.GST_VALIDATION,
                    description=f"Invalid supplier GSTIN: {supplier_gstin}",
                    plain_language_explanation="The supplier's GST number is invalid or not found in government records.",
                    suggested_fixes=["Verify and correct the supplier's GSTIN"],
                    severity=ComplianceSeverity.HIGH,
                    field_name="supplier_gstin",
                    current_value=supplier_gstin
                ))
        
        # Validate customer GSTIN (if provided)
        customer_gstin = invoice_data.get('customer_gstin')
        if customer_gstin:
            validation_result = await self.validate_gst_number(customer_gstin)
            if not validation_result.is_valid:
                issues.append(ComplianceIssue(
                    type=ComplianceType.GST_VALIDATION,
                    description=f"Invalid customer GSTIN: {customer_gstin}",
                    plain_language_explanation="The customer's GST number is invalid or not found in government records.",
                    suggested_fixes=["Verify and correct the customer's GSTIN"],
                    severity=ComplianceSeverity.MEDIUM,
                    field_name="customer_gstin",
                    current_value=customer_gstin
                ))
        
        return issues
    
    async def _verify_tax_calculations(self, invoice_data: Dict[str, Any]) -> List[ComplianceIssue]:
        """Verify tax calculations are correct"""
        issues = []
        
        try:
            expected_tax, breakdown = await self._calculate_expected_tax(invoice_data)
            actual_tax = float(invoice_data.get('tax_amount', 0))
            
            variance = abs(expected_tax - actual_tax)
            if variance > 0.01:  # More than 1 paisa difference
                issues.append(ComplianceIssue(
                    type=ComplianceType.TAX_CALCULATION,
                    description=f"Tax calculation mismatch: Expected ₹{expected_tax:.2f}, Found ₹{actual_tax:.2f}",
                    plain_language_explanation=f"The tax amount on this invoice appears to be incorrect. Based on the items and tax rates, the tax should be ₹{expected_tax:.2f} but shows ₹{actual_tax:.2f}.",
                    suggested_fixes=[
                        f"Recalculate tax amount to ₹{expected_tax:.2f}",
                        "Verify tax rates applied to each item",
                        "Check if any exemptions or special rates apply"
                    ],
                    severity=ComplianceSeverity.HIGH,
                    field_name="tax_amount",
                    current_value=str(actual_tax),
                    expected_value=str(expected_tax)
                ))
        
        except Exception as e:
            logger.error(f"Error verifying tax calculations: {e}")
            issues.append(ComplianceIssue(
                type=ComplianceType.TAX_CALCULATION,
                description="Unable to verify tax calculations",
                plain_language_explanation="There was an error checking the tax calculations on this invoice.",
                suggested_fixes=["Manually verify tax calculations"],
                severity=ComplianceSeverity.MEDIUM
            ))
        
        return issues
    
    async def _validate_place_of_supply(self, invoice_data: Dict[str, Any]) -> List[ComplianceIssue]:
        """Validate place of supply is correctly determined"""
        issues = []
        
        place_of_supply = invoice_data.get('place_of_supply')
        supplier_state = invoice_data.get('supplier_state')
        customer_state = invoice_data.get('customer_state')
        
        if not place_of_supply:
            issues.append(ComplianceIssue(
                type=ComplianceType.MISSING_FIELDS,
                description="Place of supply not specified",
                plain_language_explanation="Every GST invoice must specify the place of supply to determine if it's an intra-state or inter-state transaction.",
                suggested_fixes=["Add place of supply to the invoice"],
                severity=ComplianceSeverity.HIGH,
                field_name="place_of_supply"
            ))
        elif customer_state and place_of_supply != customer_state:
            # For B2B transactions, place of supply should match customer's state
            issues.append(ComplianceIssue(
                type=ComplianceType.GST_VALIDATION,
                description=f"Place of supply mismatch: {place_of_supply} vs customer state {customer_state}",
                plain_language_explanation="For business-to-business transactions, the place of supply should typically match the customer's registered state.",
                suggested_fixes=[f"Update place of supply to {customer_state}"],
                severity=ComplianceSeverity.MEDIUM,
                field_name="place_of_supply",
                current_value=place_of_supply,
                expected_value=customer_state
            ))
        
        return issues
    
    def _validate_gstin_format(self, gstin: str) -> bool:
        """Validate GSTIN format using regex"""
        if not gstin or len(gstin) != 15:
            return False
        
        # GSTIN format breakdown for "27AAPFU0939F1ZV":
        # Positions: 0123456789012345
        # Format:    27AAPFU0939F1ZV
        # 0-1: State code (27)
        # 2-6: First 5 chars of PAN (AAPFU)
        # 7-10: Next 4 chars of PAN (0939)
        # 11: Last char of PAN (F)
        # 12: Entity number (1)
        # 13: Always 'Z'
        # 14: Check digit (V)
        
        # Correct pattern: 2 digits + 5 letters + 4 digits + 1 letter + 1 digit/letter + Z + 1 letter/digit
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}Z[0-9A-Z]{1}$'
        return bool(re.match(pattern, gstin))
    
    async def _validate_gstin_via_api(self, gstin: str) -> GSTValidationResult:
        """Validate GSTIN via government API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.gst_api_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.gst_api_url}/taxpayerapi/search"
                payload = {'gstin': gstin}
                
                async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return GSTValidationResult(
                            gstin=gstin,
                            is_valid=data.get('valid', False),
                            business_name=data.get('tradeNam', ''),
                            status=data.get('sts', 'unknown'),
                            registration_date=datetime.fromisoformat(data.get('rgdt', '')) if data.get('rgdt') else None
                        )
                    else:
                        return GSTValidationResult(
                            gstin=gstin,
                            is_valid=False,
                            status="api_error",
                            errors=[f"API returned status {response.status}"]
                        )
        
        except asyncio.TimeoutError:
            return GSTValidationResult(
                gstin=gstin,
                is_valid=False,
                status="timeout",
                errors=["API request timed out"]
            )
        except Exception as e:
            return GSTValidationResult(
                gstin=gstin,
                is_valid=False,
                status="error",
                errors=[str(e)]
            )
    
    async def _calculate_expected_tax(self, invoice_data: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        """Calculate expected tax based on items and rates"""
        total_tax = 0.0
        breakdown = {
            'cgst': 0.0,
            'sgst': 0.0,
            'igst': 0.0,
            'items': []
        }
        
        items = invoice_data.get('items', [])
        supplier_state = invoice_data.get('supplier_state', '')
        customer_state = invoice_data.get('customer_state', '')
        
        # Determine if it's inter-state (IGST) or intra-state (CGST+SGST)
        is_inter_state = supplier_state != customer_state
        
        for item in items:
            item_value = float(item.get('taxable_value', 0))
            tax_rate = float(item.get('tax_rate', self.gst_rates['standard']))
            
            item_tax = item_value * (tax_rate / 100)
            total_tax += item_tax
            
            if is_inter_state:
                breakdown['igst'] += item_tax
            else:
                # Split equally between CGST and SGST
                cgst_sgst = item_tax / 2
                breakdown['cgst'] += cgst_sgst
                breakdown['sgst'] += cgst_sgst
            
            breakdown['items'].append({
                'name': item.get('name', ''),
                'taxable_value': item_value,
                'tax_rate': tax_rate,
                'tax_amount': item_tax
            })
        
        # Round to 2 decimal places
        total_tax = float(Decimal(str(total_tax)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        
        return total_tax, breakdown
    
    def _calculate_compliance_score(self, issues: List[ComplianceIssue]) -> float:
        """Calculate overall compliance score based on issues"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ComplianceSeverity.LOW: 0.1,
            ComplianceSeverity.MEDIUM: 0.3,
            ComplianceSeverity.HIGH: 0.6,
            ComplianceSeverity.CRITICAL: 1.0
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        max_possible_weight = len(issues) * 1.0  # If all were critical
        
        # Score decreases based on weighted severity
        score = max(0.0, 1.0 - (total_weight / max_possible_weight))
        return round(score, 2)
    
    def _determine_compliance_status(self, issues: List[ComplianceIssue]) -> ComplianceStatus:
        """Determine overall compliance status"""
        if not issues:
            return ComplianceStatus.COMPLIANT
        
        # Check for critical issues
        critical_issues = [i for i in issues if i.severity == ComplianceSeverity.CRITICAL]
        if critical_issues:
            return ComplianceStatus.CRITICAL_ISSUES
        
        # Check for high severity issues
        high_issues = [i for i in issues if i.severity == ComplianceSeverity.HIGH]
        if len(high_issues) >= 3:  # Multiple high severity issues
            return ComplianceStatus.CRITICAL_ISSUES
        elif high_issues:
            return ComplianceStatus.ISSUES_FOUND
        
        # Only medium/low issues
        return ComplianceStatus.ISSUES_FOUND
    
    # Issue tracking and resolution methods
    
    async def resolve_compliance_issue(
        self, 
        issue_id: str, 
        resolution_action: str, 
        resolution_notes: Optional[str] = None,
        resolved_by: str = None
    ) -> Dict[str, Any]:
        """Mark a compliance issue as resolved"""
        logger.info(f"Resolving compliance issue: {issue_id}")
        
        try:
            # Update issue status in database
            update_query = """
            UPDATE compliance_issues 
            SET status = 'resolved',
                resolution_action = %s,
                resolution_notes = %s,
                resolved_by = %s,
                resolved_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
            RETURNING *
            """
            
            result = await self.db.fetch_one(
                update_query, 
                (resolution_action, resolution_notes, resolved_by, issue_id)
            )
            
            if not result:
                raise ValueError(f"Compliance issue {issue_id} not found")
            
            # Log resolution activity
            await self._log_compliance_activity(
                business_id=result['business_id'],
                activity_type="issue_resolved",
                details={
                    "issue_id": issue_id,
                    "resolution_action": resolution_action,
                    "resolved_by": resolved_by
                }
            )
            
            return {
                "success": True,
                "message": "Compliance issue resolved successfully",
                "issue_id": issue_id,
                "resolved_at": result['resolved_at']
            }
            
        except Exception as e:
            logger.error(f"Error resolving compliance issue {issue_id}: {e}")
            raise
    
    async def get_compliance_tracking(self, business_id: str) -> Dict[str, Any]:
        """Get compliance tracking summary for a business"""
        logger.info(f"Getting compliance tracking for business: {business_id}")
        
        try:
            # Get issue statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_issues,
                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_issues,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as pending_issues,
                COUNT(CASE WHEN severity = 'critical' AND status = 'active' THEN 1 END) as critical_issues,
                AVG(CASE WHEN status = 'resolved' THEN 1.0 ELSE 0.0 END) as resolution_rate
            FROM compliance_issues 
            WHERE business_id = %s
            """
            
            stats_result = await self.db.fetch_one(stats_query, (business_id,))
            
            # Get issues by type
            type_query = """
            SELECT issue_type, COUNT(*) as count
            FROM compliance_issues 
            WHERE business_id = %s AND status = 'active'
            GROUP BY issue_type
            """
            
            type_results = await self.db.fetch_all(type_query, (business_id,))
            issues_by_type = {row['issue_type']: row['count'] for row in type_results}
            
            # Calculate compliance score
            total_issues = stats_result['total_issues'] or 0
            resolved_issues = stats_result['resolved_issues'] or 0
            critical_issues = stats_result['critical_issues'] or 0
            
            # Score calculation: base score reduced by active issues, especially critical ones
            base_score = 1.0
            if total_issues > 0:
                pending_issues = total_issues - resolved_issues
                score_reduction = (pending_issues * 0.1) + (critical_issues * 0.3)
                compliance_score = max(0.0, base_score - score_reduction)
            else:
                compliance_score = 1.0
            
            return {
                "business_id": business_id,
                "total_issues": total_issues,
                "resolved_issues": resolved_issues,
                "pending_issues": stats_result['pending_issues'] or 0,
                "critical_issues": critical_issues,
                "compliance_score": round(compliance_score, 2),
                "last_updated": datetime.utcnow(),
                "issues_by_type": issues_by_type,
                "resolution_rate": round(stats_result['resolution_rate'] or 0.0, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance tracking: {e}")
            raise
    
    # Automated reminder system methods
    
    async def create_compliance_reminder(
        self,
        business_id: str,
        reminder_type: str,
        due_date: datetime,
        description: str,
        priority: str = "medium",
        recurring: bool = False,
        recurring_interval_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create automated compliance reminder"""
        logger.info(f"Creating compliance reminder for business: {business_id}")
        
        try:
            # Insert reminder into database
            insert_query = """
            INSERT INTO compliance_reminders 
            (business_id, reminder_type, due_date, description, priority, 
             recurring, recurring_interval_days, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', NOW())
            RETURNING *
            """
            
            result = await self.db.fetch_one(
                insert_query,
                (business_id, reminder_type, due_date, description, priority, 
                 recurring, recurring_interval_days)
            )
            
            # Calculate next reminder date for recurring reminders
            next_reminder = None
            if recurring and recurring_interval_days:
                next_reminder = due_date + timedelta(days=recurring_interval_days)
            
            # Log reminder creation
            await self._log_compliance_activity(
                business_id=business_id,
                activity_type="reminder_created",
                details={
                    "reminder_id": str(result['id']),
                    "reminder_type": reminder_type,
                    "due_date": due_date.isoformat()
                }
            )
            
            return {
                "reminder_id": str(result['id']),
                "business_id": business_id,
                "reminder_type": reminder_type,
                "due_date": due_date,
                "description": description,
                "priority": priority,
                "status": "active",
                "created_at": result['created_at'],
                "next_reminder": next_reminder
            }
            
        except Exception as e:
            logger.error(f"Error creating compliance reminder: {e}")
            raise
    
    async def get_compliance_reminders(
        self,
        business_id: str,
        status: Optional[str] = None,
        reminder_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get compliance reminders for a business"""
        logger.info(f"Getting compliance reminders for business: {business_id}")
        
        try:
            # Build query with optional filters
            query = """
            SELECT * FROM compliance_reminders 
            WHERE business_id = %s
            """
            params = [business_id]
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            if reminder_type:
                query += " AND reminder_type = %s"
                params.append(reminder_type)
            
            query += " ORDER BY due_date ASC"
            
            results = await self.db.fetch_all(query, params)
            
            reminders = []
            for row in results:
                reminder = dict(row)
                
                # Calculate next reminder for recurring reminders
                if reminder['recurring'] and reminder['recurring_interval_days']:
                    reminder['next_reminder'] = reminder['due_date'] + timedelta(
                        days=reminder['recurring_interval_days']
                    )
                
                reminders.append(reminder)
            
            return reminders
            
        except Exception as e:
            logger.error(f"Error getting compliance reminders: {e}")
            raise
    
    async def complete_compliance_reminder(
        self,
        reminder_id: str,
        completion_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark a compliance reminder as completed"""
        logger.info(f"Completing compliance reminder: {reminder_id}")
        
        try:
            # Get reminder details first
            get_query = "SELECT * FROM compliance_reminders WHERE id = %s"
            reminder = await self.db.fetch_one(get_query, (reminder_id,))
            
            if not reminder:
                raise ValueError(f"Reminder {reminder_id} not found")
            
            # Update reminder status
            update_query = """
            UPDATE compliance_reminders 
            SET status = 'completed',
                completion_notes = %s,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
            """
            
            await self.db.execute(update_query, (completion_notes, reminder_id))
            
            # Create next occurrence for recurring reminders
            next_reminder_id = None
            if reminder['recurring'] and reminder['recurring_interval_days']:
                next_due_date = reminder['due_date'] + timedelta(
                    days=reminder['recurring_interval_days']
                )
                
                next_reminder = await self.create_compliance_reminder(
                    business_id=reminder['business_id'],
                    reminder_type=reminder['reminder_type'],
                    due_date=next_due_date,
                    description=reminder['description'],
                    priority=reminder['priority'],
                    recurring=True,
                    recurring_interval_days=reminder['recurring_interval_days']
                )
                next_reminder_id = next_reminder['reminder_id']
            
            # Log completion
            await self._log_compliance_activity(
                business_id=reminder['business_id'],
                activity_type="reminder_completed",
                details={
                    "reminder_id": reminder_id,
                    "completion_notes": completion_notes,
                    "next_reminder_id": next_reminder_id
                }
            )
            
            return {
                "success": True,
                "message": "Reminder completed successfully",
                "reminder_id": reminder_id,
                "next_reminder_id": next_reminder_id,
                "completed_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error completing compliance reminder: {e}")
            raise
    
    async def schedule_reminder_notifications(self, reminder_id: str):
        """Schedule notifications for compliance reminders"""
        logger.info(f"Scheduling notifications for reminder: {reminder_id}")
        
        try:
            # Get reminder details
            query = "SELECT * FROM compliance_reminders WHERE id = %s"
            reminder = await self.db.fetch_one(query, (reminder_id,))
            
            if not reminder:
                logger.warning(f"Reminder {reminder_id} not found for notification scheduling")
                return
            
            # Calculate notification schedule based on priority and due date
            due_date = reminder['due_date']
            priority = reminder['priority']
            
            notification_schedule = []
            
            if priority == "high":
                # High priority: 7 days, 3 days, 1 day, day of
                notification_schedule = [7, 3, 1, 0]
            elif priority == "medium":
                # Medium priority: 3 days, 1 day, day of
                notification_schedule = [3, 1, 0]
            else:
                # Low priority: 1 day, day of
                notification_schedule = [1, 0]
            
            # Schedule notifications
            for days_before in notification_schedule:
                notification_date = due_date - timedelta(days=days_before)
                
                # Only schedule future notifications
                if notification_date > datetime.utcnow():
                    await self._schedule_notification(
                        business_id=reminder['business_id'],
                        reminder_id=reminder_id,
                        notification_date=notification_date,
                        message=self._generate_reminder_message(reminder, days_before)
                    )
            
        except Exception as e:
            logger.error(f"Error scheduling reminder notifications: {e}")
    
    # Plain language explanation methods
    
    async def generate_plain_language_explanation(
        self,
        issue_type: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate plain language explanation for compliance issues"""
        logger.info(f"Generating explanation for issue type: {issue_type}")
        
        context = context or {}
        
        explanations = {
            "gst_validation": {
                "explanation": "GST number validation ensures that the GST identification numbers (GSTIN) on your invoices are correct and registered with the government. This is required for all GST-registered businesses in India.",
                "suggested_actions": [
                    "Verify the GSTIN with your customer or supplier",
                    "Check the GSTIN format (15 characters: 2 digits + 10 characters + 1 digit + Z + 1 character)",
                    "Use the government GST portal to verify the number",
                    "Update your records with the correct GSTIN"
                ],
                "resources": [
                    "GST Portal: https://www.gst.gov.in/",
                    "GSTIN verification tool",
                    "GST helpline: 1800-103-4786"
                ],
                "severity_explanation": "Invalid GST numbers can lead to input tax credit rejection and compliance issues during GST filing."
            },
            "tax_calculation": {
                "explanation": "Tax calculation verification ensures that the GST amounts on your invoices are calculated correctly based on the applicable tax rates for different goods and services.",
                "suggested_actions": [
                    "Verify the HSN/SAC codes for your products/services",
                    "Check the applicable GST rates (5%, 12%, 18%, or 28%)",
                    "Recalculate CGST, SGST, or IGST based on place of supply",
                    "Use GST calculation tools or software for accuracy"
                ],
                "resources": [
                    "GST rate finder on GST portal",
                    "HSN/SAC code directory",
                    "GST calculation formulas and examples"
                ],
                "severity_explanation": "Incorrect tax calculations can result in short payment of taxes, penalties, and interest charges."
            },
            "missing_fields": {
                "explanation": "GST invoices must contain specific mandatory fields as per GST rules. Missing any required field can make your invoice non-compliant and affect your input tax credit claims.",
                "suggested_actions": [
                    "Include all mandatory fields: Invoice number, date, GSTIN, place of supply",
                    "Add item details with HSN/SAC codes",
                    "Include taxable value, tax amount, and total amount",
                    "Review GST invoice format requirements"
                ],
                "resources": [
                    "GST invoice format guidelines",
                    "Mandatory fields checklist",
                    "Sample GST invoice templates"
                ],
                "severity_explanation": "Incomplete invoices may not be accepted for input tax credit and can cause issues during GST audits."
            },
            "deadline_warning": {
                "explanation": "GST compliance has specific deadlines for filing returns, paying taxes, and other regulatory requirements. Missing these deadlines can result in penalties and interest charges.",
                "suggested_actions": [
                    "Mark important GST deadlines in your calendar",
                    "Set up automated reminders for filing and payment dates",
                    "Prepare documents and data in advance",
                    "Consider using GST software for timely compliance"
                ],
                "resources": [
                    "GST compliance calendar",
                    "Due date calculator",
                    "Late filing penalty calculator"
                ],
                "severity_explanation": "Late filing or payment can result in penalties of ₹200 per day per return and interest charges on outstanding tax amounts."
            }
        }
        
        if issue_type not in explanations:
            return {
                "explanation": f"This is a {issue_type.replace('_', ' ')} compliance issue that requires attention.",
                "suggested_actions": ["Review the issue details and take appropriate corrective action"],
                "resources": [],
                "severity_explanation": "Please address this issue to maintain compliance."
            }
        
        explanation = explanations[issue_type].copy()
        
        # Customize explanation based on context
        if context:
            if "field_name" in context:
                explanation["explanation"] += f" The specific field '{context['field_name']}' needs attention."
            
            if "current_value" in context and "expected_value" in context:
                explanation["suggested_actions"].insert(0, 
                    f"Change '{context['current_value']}' to '{context['expected_value']}'")
        
        return explanation
    
    async def get_upcoming_deadlines(
        self,
        business_id: str,
        days_ahead: int = 30
    ) -> List[Dict[str, Any]]:
        """Get upcoming compliance deadlines for a business"""
        logger.info(f"Getting upcoming deadlines for business: {business_id}")
        
        try:
            # Get business registration date and GST details
            business_query = """
            SELECT gstin, registration_date, business_type, state
            FROM businesses 
            WHERE id = %s
            """
            business = await self.db.fetch_one(business_query, (business_id,))
            
            if not business:
                return []
            
            # Calculate standard GST deadlines
            deadlines = []
            current_date = datetime.utcnow().date()
            end_date = current_date + timedelta(days=days_ahead)
            
            # Monthly GSTR-1 deadline (11th of next month)
            for month_offset in range(3):  # Check next 3 months
                target_date = current_date.replace(day=1) + timedelta(days=32 * month_offset)
                deadline_date = target_date.replace(day=11)
                
                if current_date <= deadline_date <= end_date:
                    deadlines.append({
                        "type": "gstr1_filing",
                        "description": f"GSTR-1 filing for {target_date.strftime('%B %Y')}",
                        "due_date": deadline_date,
                        "priority": "high",
                        "penalty_info": "Late fee: ₹200 per day",
                        "days_remaining": (deadline_date - current_date).days
                    })
            
            # Monthly GSTR-3B deadline (20th of next month)
            for month_offset in range(3):
                target_date = current_date.replace(day=1) + timedelta(days=32 * month_offset)
                deadline_date = target_date.replace(day=20)
                
                if current_date <= deadline_date <= end_date:
                    deadlines.append({
                        "type": "gstr3b_filing",
                        "description": f"GSTR-3B filing for {target_date.strftime('%B %Y')}",
                        "due_date": deadline_date,
                        "priority": "high",
                        "penalty_info": "Late fee: ₹200 per day + interest on tax liability",
                        "days_remaining": (deadline_date - current_date).days
                    })
            
            # Add custom reminders from database
            custom_reminders_query = """
            SELECT reminder_type, description, due_date, priority
            FROM compliance_reminders
            WHERE business_id = %s 
            AND status = 'active'
            AND due_date BETWEEN %s AND %s
            ORDER BY due_date
            """
            
            custom_reminders = await self.db.fetch_all(
                custom_reminders_query,
                (business_id, current_date, end_date)
            )
            
            for reminder in custom_reminders:
                deadlines.append({
                    "type": reminder['reminder_type'],
                    "description": reminder['description'],
                    "due_date": reminder['due_date'],
                    "priority": reminder['priority'],
                    "penalty_info": "Custom reminder - check specific requirements",
                    "days_remaining": (reminder['due_date'] - current_date).days
                })
            
            # Sort by due date
            deadlines.sort(key=lambda x: x['due_date'])
            
            return deadlines
            
        except Exception as e:
            logger.error(f"Error getting upcoming deadlines: {e}")
            raise
    
    # Bulk processing methods
    
    async def start_bulk_compliance_check(
        self,
        business_id: str,
        invoice_ids: List[str]
    ) -> str:
        """Start bulk compliance checking job"""
        import uuid
        job_id = str(uuid.uuid4())
        
        logger.info(f"Starting bulk compliance check job: {job_id}")
        
        try:
            # Create job record
            insert_query = """
            INSERT INTO bulk_compliance_jobs 
            (id, business_id, invoice_ids, status, created_at)
            VALUES (%s, %s, %s, 'started', NOW())
            """
            
            await self.db.execute(
                insert_query,
                (job_id, business_id, invoice_ids)
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting bulk compliance check: {e}")
            raise
    
    async def process_bulk_compliance_check(
        self,
        job_id: str,
        business_id: str,
        invoice_ids: List[str]
    ):
        """Process bulk compliance check in background"""
        logger.info(f"Processing bulk compliance check job: {job_id}")
        
        try:
            # Update job status to processing
            await self.db.execute(
                "UPDATE bulk_compliance_jobs SET status = 'processing', started_at = NOW() WHERE id = %s",
                (job_id,)
            )
            
            results = []
            processed = 0
            
            for invoice_id in invoice_ids:
                try:
                    # Check compliance for each invoice
                    compliance_result = await self.check_compliance(invoice_id)
                    results.append({
                        "invoice_id": invoice_id,
                        "status": "completed",
                        "issues_count": len(compliance_result.issues),
                        "compliance_score": compliance_result.compliance_score
                    })
                    processed += 1
                    
                    # Update progress
                    progress = (processed / len(invoice_ids)) * 100
                    await self.db.execute(
                        "UPDATE bulk_compliance_jobs SET progress = %s WHERE id = %s",
                        (progress, job_id)
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing invoice {invoice_id}: {e}")
                    results.append({
                        "invoice_id": invoice_id,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Update job with final results
            await self.db.execute(
                """UPDATE bulk_compliance_jobs 
                   SET status = 'completed', results = %s, completed_at = NOW() 
                   WHERE id = %s""",
                (results, job_id)
            )
            
        except Exception as e:
            logger.error(f"Error processing bulk compliance check: {e}")
            await self.db.execute(
                "UPDATE bulk_compliance_jobs SET status = 'failed', error = %s WHERE id = %s",
                (str(e), job_id)
            )
    
    async def get_bulk_check_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of bulk compliance check job"""
        try:
            query = "SELECT * FROM bulk_compliance_jobs WHERE id = %s"
            result = await self.db.fetch_one(query, (job_id,))
            
            if not result:
                raise ValueError(f"Job {job_id} not found")
            
            return dict(result)
            
        except Exception as e:
            logger.error(f"Error getting bulk check status: {e}")
            raise
    
    # Helper methods
    
    async def _log_compliance_activity(
        self,
        business_id: str,
        activity_type: str,
        details: Dict[str, Any]
    ):
        """Log compliance activity for audit trail"""
        try:
            insert_query = """
            INSERT INTO compliance_activity_log 
            (business_id, activity_type, details, created_at)
            VALUES (%s, %s, %s, NOW())
            """
            
            await self.db.execute(insert_query, (business_id, activity_type, details))
            
        except Exception as e:
            logger.error(f"Error logging compliance activity: {e}")
    
    async def _schedule_notification(
        self,
        business_id: str,
        reminder_id: str,
        notification_date: datetime,
        message: str
    ):
        """Schedule a notification for compliance reminder"""
        try:
            insert_query = """
            INSERT INTO scheduled_notifications 
            (business_id, reminder_id, notification_date, message, status, created_at)
            VALUES (%s, %s, %s, %s, 'scheduled', NOW())
            """
            
            await self.db.execute(
                insert_query,
                (business_id, reminder_id, notification_date, message)
            )
            
        except Exception as e:
            logger.error(f"Error scheduling notification: {e}")
    
    def _generate_reminder_message(self, reminder: Dict[str, Any], days_before: int) -> str:
        """Generate reminder notification message"""
        reminder_type = reminder['reminder_type']
        description = reminder['description']
        
        if days_before == 0:
            return f"⚠️ Due Today: {description}"
        elif days_before == 1:
            return f"⏰ Due Tomorrow: {description}"
        else:
            return f"📅 Due in {days_before} days: {description}"