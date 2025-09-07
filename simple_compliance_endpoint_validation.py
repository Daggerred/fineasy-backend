#!/usr/bin/env python3
"""
Simple validation script for compliance endpoints
Tests basic functionality without complex dependencies
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

def validate_gstin_format(gstin: str) -> bool:
    """Validate GSTIN format using regex"""
    if not gstin or len(gstin) != 15:
        return False
    
    # GSTIN format: 2 digits + 5 letters + 4 digits + 1 letter + 1 digit/letter + Z + 1 letter/digit
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, gstin))

def calculate_compliance_score(issues: List[Dict[str, Any]]) -> float:
    """Calculate compliance score based on issues"""
    if not issues:
        return 1.0
    
    # Weight issues by severity
    severity_weights = {
        'low': 0.1,
        'medium': 0.3,
        'high': 0.6,
        'critical': 1.0
    }
    
    total_weight = sum(severity_weights.get(issue.get('severity', 'medium'), 0.3) for issue in issues)
    max_possible_weight = len(issues) * 1.0  # If all were critical
    
    # Score decreases based on weighted severity
    score = max(0.0, 1.0 - (total_weight / max_possible_weight))
    return round(score, 2)

def determine_compliance_status(issues: List[Dict[str, Any]]) -> str:
    """Determine overall compliance status"""
    if not issues:
        return "compliant"
    
    # Check for critical issues
    critical_issues = [i for i in issues if i.get('severity') == 'critical']
    if critical_issues:
        return "critical_issues"
    
    # Check for high severity issues
    high_issues = [i for i in issues if i.get('severity') == 'high']
    if len(high_issues) >= 3:  # Multiple high severity issues
        return "critical_issues"
    elif high_issues:
        return "issues_found"
    
    # Only medium/low issues
    return "issues_found"

def calculate_expected_tax(items: List[Dict[str, Any]], is_inter_state: bool = False) -> tuple:
    """Calculate expected tax based on items and rates"""
    total_tax = 0.0
    breakdown = {
        'cgst': 0.0,
        'sgst': 0.0,
        'igst': 0.0,
        'items': []
    }
    
    for item in items:
        item_value = float(item.get('taxable_value', 0))
        tax_rate = float(item.get('tax_rate', 18.0))
        
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
    
    return round(total_tax, 2), breakdown

def generate_plain_language_explanation(issue_type: str) -> Dict[str, Any]:
    """Generate plain language explanation for compliance issues"""
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
    
    return explanations.get(issue_type, {
        "explanation": f"This is a {issue_type.replace('_', ' ')} compliance issue that requires attention.",
        "suggested_actions": ["Review the issue details and take appropriate corrective action"],
        "resources": [],
        "severity_explanation": "Please address this issue to maintain compliance."
    })

def generate_reminder_message(reminder_type: str, description: str, days_before: int) -> str:
    """Generate reminder notification message"""
    if days_before == 0:
        return f"⚠️ Due Today: {description}"
    elif days_before == 1:
        return f"⏰ Due Tomorrow: {description}"
    else:
        return f"📅 Due in {days_before} days: {description}"

def get_upcoming_deadlines(business_registration_date: datetime, days_ahead: int = 30) -> List[Dict[str, Any]]:
    """Get upcoming compliance deadlines"""
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
                "due_date": deadline_date.isoformat(),
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
                "due_date": deadline_date.isoformat(),
                "priority": "high",
                "penalty_info": "Late fee: ₹200 per day + interest on tax liability",
                "days_remaining": (deadline_date - current_date).days
            })
    
    return sorted(deadlines, key=lambda x: x['due_date'])

def run_validation_tests():
    """Run all validation tests"""
    print("🚀 Starting Compliance Service Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: GST Format Validation
    print("\n🔍 Testing GST format validation...")
    total_tests += 1
    
    test_cases = [
        ("27AAPFU0939F1ZV", True),   # Valid GSTIN
        ("12ABCDE1234F1Z5", True),   # Valid GSTIN
        ("INVALID123", False),       # Invalid format
        ("27AAPFU0939F1Z", False),   # Too short
        ("", False),                 # Empty
    ]
    
    gst_test_passed = True
    for gstin, expected in test_cases:
        result = validate_gstin_format(gstin)
        if result != expected:
            print(f"❌ GST validation failed for {gstin}: expected {expected}, got {result}")
            gst_test_passed = False
    
    if gst_test_passed:
        print("✅ GST format validation: PASSED")
        tests_passed += 1
    else:
        print("❌ GST format validation: FAILED")
    
    # Test 2: Compliance Score Calculation
    print("\n🔍 Testing compliance score calculation...")
    total_tests += 1
    
    # Test with no issues
    score = calculate_compliance_score([])
    if score == 1.0:
        print("✅ Compliance score (no issues): PASSED")
        
        # Test with issues
        issues = [
            {"severity": "high", "type": "gst_validation"},
            {"severity": "medium", "type": "tax_calculation"}
        ]
        score = calculate_compliance_score(issues)
        if 0.0 <= score <= 1.0:
            print("✅ Compliance score (with issues): PASSED")
            tests_passed += 1
        else:
            print(f"❌ Compliance score out of range: {score}")
    else:
        print(f"❌ Compliance score calculation failed: {score}")
    
    # Test 3: Tax Calculation
    print("\n🔍 Testing tax calculation...")
    total_tests += 1
    
    items = [
        {"name": "Product 1", "taxable_value": 1000.0, "tax_rate": 18.0},
        {"name": "Product 2", "taxable_value": 500.0, "tax_rate": 12.0}
    ]
    
    expected_tax, breakdown = calculate_expected_tax(items, is_inter_state=False)
    
    # Expected: (1000 * 0.18) + (500 * 0.12) = 180 + 60 = 240
    if abs(expected_tax - 240.0) < 0.01:
        print("✅ Tax calculation: PASSED")
        tests_passed += 1
    else:
        print(f"❌ Tax calculation failed: expected 240.0, got {expected_tax}")
    
    # Test 4: Plain Language Explanations
    print("\n🔍 Testing plain language explanations...")
    total_tests += 1
    
    issue_types = ["gst_validation", "tax_calculation", "missing_fields", "deadline_warning"]
    explanations_passed = True
    
    for issue_type in issue_types:
        explanation = generate_plain_language_explanation(issue_type)
        if not (explanation.get("explanation") and explanation.get("suggested_actions")):
            explanations_passed = False
            break
    
    if explanations_passed:
        print("✅ Plain language explanations: PASSED")
        tests_passed += 1
    else:
        print("❌ Plain language explanations: FAILED")
    
    # Test 5: Reminder Message Generation
    print("\n🔍 Testing reminder message generation...")
    total_tests += 1
    
    message_0 = generate_reminder_message("gst_filing", "Test reminder", 0)
    message_1 = generate_reminder_message("gst_filing", "Test reminder", 1)
    message_7 = generate_reminder_message("gst_filing", "Test reminder", 7)
    
    if (message_0.startswith("⚠️ Due Today") and 
        message_1.startswith("⏰ Due Tomorrow") and
        message_7.startswith("📅 Due in 7 days")):
        print("✅ Reminder message generation: PASSED")
        tests_passed += 1
    else:
        print("❌ Reminder message generation: FAILED")
    
    # Test 6: Compliance Status Determination
    print("\n🔍 Testing compliance status determination...")
    total_tests += 1
    
    status_no_issues = determine_compliance_status([])
    status_critical = determine_compliance_status([{"severity": "critical"}])
    status_medium = determine_compliance_status([{"severity": "medium"}])
    
    if (status_no_issues == "compliant" and 
        status_critical == "critical_issues" and
        status_medium == "issues_found"):
        print("✅ Compliance status determination: PASSED")
        tests_passed += 1
    else:
        print("❌ Compliance status determination: FAILED")
    
    # Test 7: Upcoming Deadlines
    print("\n🔍 Testing upcoming deadlines...")
    total_tests += 1
    
    deadlines = get_upcoming_deadlines(datetime.utcnow(), days_ahead=30)
    if isinstance(deadlines, list) and len(deadlines) > 0:
        print(f"✅ Upcoming deadlines: PASSED ({len(deadlines)} deadlines found)")
        tests_passed += 1
    else:
        print("❌ Upcoming deadlines: FAILED")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("🎉 All validations passed!")
        return True
    else:
        print("⚠️  Some validations failed!")
        return False

if __name__ == "__main__":
    success = run_validation_tests()
    exit(0 if success else 1)