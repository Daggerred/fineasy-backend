# Compliance API Endpoints Documentation

## Overview

This document describes the comprehensive compliance API endpoints implemented for the AI Business Intelligence system. These endpoints provide GST compliance checking, issue tracking, automated reminders, and plain language explanations for Indian businesses.

## Features Implemented

### ✅ Core Compliance Features
- **GST Number Validation**: Format validation and government API verification
- **Tax Calculation Verification**: Automatic verification of GST calculations
- **Invoice Field Completeness**: Check for mandatory GST invoice fields
- **Place of Supply Validation**: Verify correct place of supply determination

### ✅ Issue Tracking and Resolution
- **Compliance Issue Tracking**: Track and manage compliance issues
- **Issue Resolution Workflow**: Mark issues as resolved with audit trail
- **Compliance Scoring**: Calculate overall compliance scores
- **Issue Analytics**: Get compliance statistics and trends

### ✅ Automated Reminder System
- **Compliance Reminders**: Create automated reminders for GST deadlines
- **Recurring Reminders**: Support for recurring compliance tasks
- **Notification Scheduling**: Intelligent notification scheduling based on priority
- **Deadline Management**: Track upcoming compliance deadlines

### ✅ Plain Language Explanations
- **User-Friendly Explanations**: Convert technical compliance issues to plain language
- **Contextual Guidance**: Provide specific guidance based on issue context
- **Resource Links**: Include relevant resources and help links
- **Severity Explanations**: Explain the impact of different issue severities

### ✅ Bulk Processing
- **Bulk Compliance Checks**: Process multiple invoices simultaneously
- **Background Processing**: Handle large batches in background tasks
- **Progress Tracking**: Monitor bulk processing progress
- **Result Aggregation**: Consolidated results for bulk operations

## API Endpoints

### 1. Health Check
```http
GET /api/v1/compliance/health
```

**Response:**
```json
{
  "service": "compliance",
  "status": "healthy",
  "gst_api_configured": true,
  "features": {
    "gst_validation": true,
    "tax_verification": true,
    "field_validation": true,
    "place_of_supply_validation": true,
    "issue_tracking": true,
    "automated_reminders": true,
    "plain_language_explanations": true,
    "bulk_processing": true
  }
}
```

### 2. Basic Compliance Check
```http
POST /api/v1/compliance/check
```

**Request Body:**
```json
{
  "invoice_id": "uuid",
  "business_id": "uuid"
}
```

**Response:**
```json
{
  "success": true,
  "invoice_id": "uuid",
  "issues": [
    {
      "id": "uuid",
      "type": "gst_validation",
      "description": "Invalid supplier GSTIN",
      "plain_language_explanation": "The supplier's GST number is invalid...",
      "suggested_fixes": ["Verify and correct the supplier's GSTIN"],
      "severity": "high",
      "field_name": "supplier_gstin",
      "current_value": "INVALID123",
      "expected_value": null
    }
  ],
  "overall_status": "issues_found",
  "compliance_score": 0.75,
  "last_checked": "2024-01-15T10:30:00Z"
}
```

### 3. GST Number Validation
```http
POST /api/v1/compliance/gst/validate
```

**Request Body:**
```json
{
  "gstin": "27AAPFU0939F1ZV"
}
```

**Response:**
```json
{
  "gstin": "27AAPFU0939F1ZV",
  "is_valid": true,
  "business_name": "Example Business Pvt Ltd",
  "status": "active",
  "registration_date": "2020-01-15T00:00:00Z",
  "errors": []
}
```

### 4. Tax Calculation Verification
```http
POST /api/v1/compliance/tax/verify?invoice_id=uuid
```

**Response:**
```json
{
  "invoice_id": "uuid",
  "calculated_tax": 180.00,
  "expected_tax": 180.00,
  "variance": 0.00,
  "is_correct": true,
  "breakdown": {
    "cgst": 90.00,
    "sgst": 90.00,
    "igst": 0.00,
    "items": [
      {
        "name": "Product A",
        "taxable_value": 1000.00,
        "tax_rate": 18.0,
        "tax_amount": 180.00
      }
    ]
  }
}
```

### 5. Get Compliance Issues
```http
GET /api/v1/compliance/issues/{invoice_id}
```

**Response:**
```json
[
  {
    "id": "uuid",
    "type": "missing_fields",
    "description": "Missing required field: place_of_supply",
    "plain_language_explanation": "This invoice is missing the place of supply field...",
    "suggested_fixes": ["Add the place of supply to the invoice"],
    "severity": "high",
    "field_name": "place_of_supply"
  }
]
```

### 6. Resolve Compliance Issue
```http
POST /api/v1/compliance/issues/{issue_id}/resolve
```

**Request Body:**
```json
{
  "issue_id": "uuid",
  "resolution_action": "Updated place of supply field",
  "resolution_notes": "Added correct place of supply based on customer location",
  "resolved_by": "user_uuid"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Compliance issue resolved successfully",
  "issue_id": "uuid",
  "resolved_at": "2024-01-15T10:30:00Z"
}
```

### 7. Get Compliance Tracking
```http
GET /api/v1/compliance/tracking/{business_id}
```

**Response:**
```json
{
  "business_id": "uuid",
  "total_issues": 15,
  "resolved_issues": 10,
  "pending_issues": 5,
  "critical_issues": 1,
  "compliance_score": 0.85,
  "last_updated": "2024-01-15T10:30:00Z",
  "issues_by_type": {
    "gst_validation": 3,
    "tax_calculation": 1,
    "missing_fields": 1
  },
  "resolution_rate": 0.67
}
```

### 8. Create Compliance Reminder
```http
POST /api/v1/compliance/reminders
```

**Request Body:**
```json
{
  "business_id": "uuid",
  "reminder_type": "gst_filing",
  "due_date": "2024-02-11T00:00:00Z",
  "description": "GSTR-1 filing for January 2024",
  "priority": "high",
  "recurring": true,
  "recurring_interval_days": 30
}
```

**Response:**
```json
{
  "reminder_id": "uuid",
  "business_id": "uuid",
  "reminder_type": "gst_filing",
  "due_date": "2024-02-11T00:00:00Z",
  "description": "GSTR-1 filing for January 2024",
  "priority": "high",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "next_reminder": "2024-03-11T00:00:00Z"
}
```

### 9. Get Compliance Reminders
```http
GET /api/v1/compliance/reminders/{business_id}?status=active&reminder_type=gst_filing
```

**Response:**
```json
[
  {
    "reminder_id": "uuid",
    "business_id": "uuid",
    "reminder_type": "gst_filing",
    "due_date": "2024-02-11T00:00:00Z",
    "description": "GSTR-1 filing for January 2024",
    "priority": "high",
    "status": "active",
    "created_at": "2024-01-15T10:30:00Z"
  }
]
```

### 10. Complete Compliance Reminder
```http
PUT /api/v1/compliance/reminders/{reminder_id}/complete?completion_notes=Filed successfully
```

**Response:**
```json
{
  "success": true,
  "message": "Reminder completed successfully",
  "reminder_id": "uuid",
  "next_reminder_id": "uuid",
  "completed_at": "2024-01-15T10:30:00Z"
}
```

### 11. Get Plain Language Explanation
```http
GET /api/v1/compliance/explanations/{issue_type}
```

**Response:**
```json
{
  "issue_type": "gst_validation",
  "explanation": "GST number validation ensures that the GST identification numbers (GSTIN) on your invoices are correct and registered with the government...",
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
}
```

### 12. Get Upcoming Deadlines
```http
GET /api/v1/compliance/deadlines/{business_id}?days_ahead=30
```

**Response:**
```json
{
  "business_id": "uuid",
  "deadlines": [
    {
      "type": "gstr1_filing",
      "description": "GSTR-1 filing for January 2024",
      "due_date": "2024-02-11",
      "priority": "high",
      "penalty_info": "Late fee: ₹200 per day",
      "days_remaining": 5
    },
    {
      "type": "gstr3b_filing",
      "description": "GSTR-3B filing for January 2024",
      "due_date": "2024-02-20",
      "priority": "high",
      "penalty_info": "Late fee: ₹200 per day + interest on tax liability",
      "days_remaining": 14
    }
  ],
  "total_deadlines": 2,
  "critical_deadlines": 1,
  "generated_at": "2024-01-15T10:30:00Z"
}
```

### 13. Bulk Compliance Check
```http
POST /api/v1/compliance/bulk-check?business_id=uuid
```

**Request Body:**
```json
["invoice_id_1", "invoice_id_2", "invoice_id_3"]
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "started",
  "total_invoices": 3,
  "estimated_completion": "2024-01-15T10:36:00Z"
}
```

### 14. Get Bulk Check Status
```http
GET /api/v1/compliance/bulk-check/{job_id}/status
```

**Response:**
```json
{
  "id": "uuid",
  "business_id": "uuid",
  "status": "completed",
  "progress": 100.0,
  "results": [
    {
      "invoice_id": "invoice_id_1",
      "status": "completed",
      "issues_count": 2,
      "compliance_score": 0.8
    },
    {
      "invoice_id": "invoice_id_2",
      "status": "completed",
      "issues_count": 0,
      "compliance_score": 1.0
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z"
}
```

## Database Schema

### New Tables Added

1. **compliance_issues** - Tracks compliance issues and their resolution
2. **compliance_reminders** - Automated compliance reminders
3. **scheduled_notifications** - Scheduled notifications for reminders
4. **compliance_activity_log** - Audit trail for compliance activities
5. **bulk_compliance_jobs** - Background jobs for bulk processing

### Key Features

- **Row Level Security (RLS)** enabled on all tables
- **Automatic timestamps** with triggers
- **Comprehensive indexing** for performance
- **Utility functions** for common operations
- **Data cleanup functions** for maintenance

## Authentication

All endpoints require Bearer token authentication:
```http
Authorization: Bearer <jwt_token>
```

## Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid request parameters"
}
```

**401 Unauthorized:**
```json
{
  "detail": "Authentication required"
}
```

**404 Not Found:**
```json
{
  "detail": "Resource not found"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error: <error_message>"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Compliance checking is currently disabled"
}
```

## Configuration

### Environment Variables

```env
# Compliance Feature Toggle
COMPLIANCE_CHECKING_ENABLED=true

# GST API Configuration
GST_API_URL=https://api.gst.gov.in
GST_API_KEY=your_gst_api_key

# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
```

## Usage Examples

### Python Client Example

```python
import httpx
import asyncio

async def check_invoice_compliance(invoice_id: str, business_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/compliance/check",
            json={
                "invoice_id": invoice_id,
                "business_id": business_id
            },
            headers={"Authorization": "Bearer your_token"}
        )
        return response.json()

# Usage
result = asyncio.run(check_invoice_compliance("invoice_id", "business_id"))
print(f"Compliance Score: {result['compliance_score']}")
```

### JavaScript/Flutter Example

```javascript
async function validateGSTNumber(gstin) {
  const response = await fetch('/api/v1/compliance/gst/validate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer your_token'
    },
    body: JSON.stringify({ gstin })
  });
  
  return await response.json();
}

// Usage
const result = await validateGSTNumber('27AAPFU0939F1ZV');
console.log(`GST Valid: ${result.is_valid}`);
```

## Testing

### Validation Scripts

1. **Core Functionality Test:**
   ```bash
   python3 simple_compliance_endpoint_validation.py
   ```

2. **API Endpoint Test:**
   ```bash
   python3 test_api_endpoints.py
   ```

3. **Comprehensive Integration Test:**
   ```bash
   python3 test_compliance_endpoints.py
   ```

## Performance Considerations

- **Caching**: GST validation results cached for 24 hours
- **Background Processing**: Bulk operations processed asynchronously
- **Database Indexing**: Optimized queries with proper indexing
- **Rate Limiting**: GST API calls rate-limited to prevent abuse

## Security Features

- **Row Level Security**: Data isolation by business
- **Audit Logging**: All compliance activities logged
- **Data Encryption**: Sensitive data encrypted at rest
- **Input Validation**: Comprehensive input validation and sanitization

## Monitoring and Logging

- **Health Checks**: Service health monitoring
- **Performance Metrics**: Processing time tracking
- **Error Logging**: Comprehensive error logging
- **Usage Analytics**: API usage tracking

## Future Enhancements

- **Machine Learning**: Predictive compliance issue detection
- **Advanced Analytics**: Compliance trend analysis
- **Integration**: Additional government API integrations
- **Automation**: Automated compliance filing assistance

## Support

For technical support or questions about the compliance API:
- Check the health endpoint for service status
- Review error logs for troubleshooting
- Validate configuration settings
- Test with the provided validation scripts