# Supabase Integration for AI Backend

This document describes the Supabase integration implementation for the FinEasy AI Backend.

## Overview

The AI Backend integrates with Supabase to provide:
- Secure database access for AI analysis
- Authentication middleware for API security
- Extended database schema for AI-specific data
- Comprehensive database utilities for AI operations

## Components Implemented

### 1. Database Connection Module (`app/database.py`)

**Features:**
- Supabase client initialization and management
- Connection testing and health checks
- Context manager for database sessions
- Comprehensive DatabaseManager class with AI-specific operations

**Key Classes:**
- `DatabaseManager`: Main class for all database operations
- Functions: `init_database()`, `test_connection()`, `get_supabase()`

**Operations Supported:**
- Business data retrieval
- Transaction and invoice queries with filtering
- Customer and supplier management
- AI analysis result storage and retrieval
- Fraud alert management
- Business insights storage
- ML model metadata management
- Processing logs and statistics

### 2. Authentication Middleware (`app/utils/auth.py`)

**Features:**
- JWT token verification for Supabase authentication
- User business access validation
- Role-based access control
- Rate limiting support

**Key Classes:**
- `AuthToken`: Authentication token data container
- `AuthMiddleware`: Middleware utilities for security

**Functions:**
- `get_current_user()`: Extract authenticated user from request
- `require_business_access()`: Ensure user has business access
- `verify_business_ownership()`: Validate business ownership

### 3. Database Schema Extensions (`schema/ai_extensions.sql`)

**New Tables Created:**
- `ai_analysis_results`: Stores AI analysis results with metadata
- `fraud_alerts`: Manages fraud detection alerts and their status
- `business_insights`: Stores AI-generated business insights
- `ml_models`: Metadata for machine learning models
- `ai_processing_logs`: Logs for monitoring and debugging

**Features:**
- Row Level Security (RLS) policies for data protection
- Comprehensive indexing for performance
- Automatic timestamp updates
- Data cleanup utilities
- Proper foreign key relationships

### 4. Database Utilities (`app/utils/database.py`)

**Features:**
- Data anonymization for privacy protection
- Batch operations for efficiency
- Caching mechanisms for analysis results
- Business statistics aggregation
- Processing log management

**Key Classes:**
- `AIDataUtils`: Utility class for AI-specific database operations

**Utilities:**
- Data hashing for deduplication
- Financial data anonymization
- Batch saving of analysis results
- Cache checking and management
- Business data summarization

## Configuration

### Environment Variables

Required variables in `.env`:
```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key

# Security
SECRET_KEY=your-secret-key-change-in-production

# Feature Flags
FRAUD_DETECTION_ENABLED=true
PREDICTIVE_ANALYTICS_ENABLED=false
COMPLIANCE_CHECKING_ENABLED=false
NLP_INVOICE_ENABLED=false
```

### Database Setup

1. **Run the schema extensions:**
   ```sql
   -- Execute the contents of schema/ai_extensions.sql in your Supabase SQL editor
   ```

2. **Verify tables are created:**
   - ai_analysis_results
   - fraud_alerts
   - business_insights
   - ml_models
   - ai_processing_logs

## Usage Examples

### Basic Database Operations

```python
from app.database import DatabaseManager

# Initialize database manager
db = DatabaseManager()

# Get business data
business = await db.get_business_data("business-id")

# Get transactions with filtering
transactions = await db.get_transactions(
    business_id="business-id",
    start_date="2024-01-01",
    end_date="2024-12-31",
    limit=100
)

# Save analysis result
analysis_data = {
    "business_id": "business-id",
    "analysis_type": "fraud_detection",
    "entity_id": "transaction-id",
    "entity_type": "transaction",
    "results": {"confidence": 0.85, "risk_score": 0.3},
    "confidence_score": 0.85
}
result_id = await db.save_analysis_result(analysis_data)
```

### Authentication Usage

```python
from fastapi import Depends
from app.utils.auth import get_current_user, require_business_access

@app.post("/api/v1/analyze")
async def analyze_data(
    business_id: str,
    auth: AuthToken = Depends(require_business_access)
):
    # User is authenticated and has business access
    # auth.user_id, auth.business_id are available
    pass
```

### AI Data Utilities

```python
from app.utils.database import AIDataUtils

# Anonymize sensitive data
sensitive_data = {"customer_name": "John Doe", "amount": 100}
anonymized = AIDataUtils.anonymize_financial_data(sensitive_data)

# Generate data hash for deduplication
data_hash = AIDataUtils.generate_data_hash(sensitive_data)

# Check analysis cache
cached_result = await ai_utils.check_analysis_cache(
    business_id="business-id",
    analysis_type="fraud_detection",
    entity_id="transaction-id",
    cache_hours=1
)
```

## Security Features

### Data Protection
- Row Level Security (RLS) policies ensure users only access their data
- Service role access for AI backend operations
- Data anonymization for sensitive information
- Encrypted data transmission

### Authentication
- JWT token verification with Supabase
- Business ownership validation
- Role-based access control
- Rate limiting support

### Privacy
- Sensitive data anonymization before AI processing
- Data hashing for deduplication without exposing content
- Automatic data cleanup and retention policies
- Audit logging for all AI operations

## Performance Optimizations

### Database Indexing
- Optimized indexes on frequently queried columns
- Composite indexes for complex queries
- Partial indexes for filtered queries

### Caching
- Analysis result caching to avoid reprocessing
- ML model caching for faster inference
- Redis integration for distributed caching

### Batch Operations
- Batch saving of analysis results
- Bulk data retrieval with pagination
- Efficient data aggregation queries

## Monitoring and Debugging

### Processing Logs
- Comprehensive logging of all AI operations
- Performance metrics tracking
- Error logging and debugging information
- Operation timing and resource usage

### Health Checks
- Database connection monitoring
- Service health endpoints
- Configuration validation
- Feature flag status reporting

## Testing

### Validation Script
Run the validation script to verify integration:
```bash
cd ai-backend
python3 simple_validation.py
```

### Test Coverage
- Unit tests for database operations
- Integration tests with mocked Supabase
- Authentication middleware testing
- Data utility function testing

## Deployment Considerations

### Production Setup
1. Use strong, unique SECRET_KEY
2. Configure proper CORS origins
3. Set up Redis for caching
4. Enable monitoring and logging
5. Configure backup and retention policies

### Scaling
- Connection pooling for high load
- Read replicas for analytics queries
- Horizontal scaling with load balancers
- Caching layers for frequently accessed data

## Troubleshooting

### Common Issues
1. **Database Connection Failed**: Check Supabase URL and keys
2. **Authentication Errors**: Verify JWT token format and signature
3. **Permission Denied**: Check RLS policies and user permissions
4. **Performance Issues**: Review indexes and query optimization

### Debug Tools
- Health check endpoint: `/health`
- Processing logs in `ai_processing_logs` table
- Configuration validation in startup
- Comprehensive error logging

## Requirements Satisfied

This implementation satisfies the following requirements from the task:

✅ **7.3**: Data Privacy and Security
- Encrypted data processing
- Anonymized data patterns
- Secure data storage
- Compliance with data protection regulations

✅ **7.4**: Data Privacy and Security  
- Purge processing data when AI features disabled
- Security breach detection and response
- Administrator alerts for security issues
- Data retention and cleanup policies

The Supabase integration provides a robust, secure, and scalable foundation for AI-powered business intelligence features in the FinEasy application.