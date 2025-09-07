# AI Backend Test Suite

This directory contains comprehensive test suites for all AI service components in the FinEasy application.

## Test Structure

```
tests/
├── conftest.py                           # Pytest configuration and shared fixtures
├── README.md                            # This file
├── test_fraud_detection.py             # Unit tests for fraud detection service
├── test_predictive_analytics.py        # Unit tests for predictive analytics service
├── test_nlp_invoice_comprehensive.py   # Unit tests for NLP invoice generation
├── test_compliance.py                  # Unit tests for compliance checking
├── test_ml_engine.py                   # Unit tests for ML engine
├── test_smart_notifications.py         # Unit tests for smart notifications
├── test_insights_api.py                # Integration tests for insights API
├── test_fraud_api.py                   # Integration tests for fraud API
├── test_api_integration.py             # Comprehensive API integration tests
├── test_end_to_end_workflows.py        # End-to-end workflow tests
├── test_performance_load.py            # Performance and load tests
├── test_security_privacy.py            # Security and privacy tests
└── test_database.py                    # Database integration tests
```

## Test Categories

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Files**: `test_fraud_detection.py`, `test_predictive_analytics.py`, `test_nlp_invoice_comprehensive.py`, `test_ml_engine.py`
- **Coverage**: All service classes, utility functions, and data models
- **Run with**: `pytest -m unit`

### 2. Integration Tests
- **Purpose**: Test component interactions and API endpoints
- **Files**: `test_api_integration.py`, `test_insights_api.py`, `test_fraud_api.py`
- **Coverage**: API endpoints, service integrations, database operations
- **Run with**: `pytest -m integration`

### 3. End-to-End Tests
- **Purpose**: Test complete workflows from start to finish
- **Files**: `test_end_to_end_workflows.py`
- **Coverage**: Complete business processes, user journeys
- **Run with**: `pytest tests/test_end_to_end_workflows.py`

### 4. Performance Tests
- **Purpose**: Test performance, scalability, and resource usage
- **Files**: `test_performance_load.py`
- **Coverage**: Response times, memory usage, concurrent load
- **Run with**: `pytest -m performance`

### 5. Accuracy Tests
- **Purpose**: Validate ML model accuracy and prediction quality
- **Files**: `test_prediction_accuracy.py`
- **Coverage**: Model predictions, fraud detection accuracy, insight quality
- **Run with**: `pytest -m accuracy`

## Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance

# Run with coverage report
python run_tests.py --coverage

# Run in CI mode (comprehensive)
python run_tests.py --ci
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_fraud_detection.py

# Run tests with specific markers
pytest -m "unit and not slow"
pytest -m "integration"
pytest -m "performance"

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Test Markers
- `unit`: Unit tests
- `integration`: Integration tests
- `performance`: Performance tests
- `accuracy`: Accuracy validation tests
- `slow`: Slow-running tests
- `smoke`: Quick smoke tests

## Test Configuration

### Environment Variables
Set these environment variables for testing:
```bash
export SUPABASE_URL="https://test.supabase.co"
export SUPABASE_SERVICE_KEY="test_service_key"
export REDIS_URL="redis://localhost:6379"
export ENVIRONMENT="test"
```

### Test Database
Tests use mocked database operations by default. For integration tests with real database:
1. Set up a test Supabase project
2. Configure test environment variables
3. Run tests with `--integration` flag

### Dependencies
Install test dependencies:
```bash
pip install -r requirements.txt
pip install pytest-xdist pytest-timeout pytest-mock
```

## Test Data

### Fixtures
Common test fixtures are defined in `conftest.py`:
- `sample_business_data`: Sample business information
- `sample_transactions`: Sample transaction data
- `sample_invoices`: Sample invoice data
- `sample_customers`: Sample customer data
- `sample_products`: Sample product data
- `mock_supabase_client`: Mocked Supabase client
- `mock_redis_client`: Mocked Redis client

### Test Data Generation
Tests use realistic data generation with:
- NumPy for statistical distributions
- Faker for realistic names and addresses
- Custom generators for business-specific data

## Coverage Requirements

### Minimum Coverage Targets
- **Overall**: 80%
- **Services**: 85%
- **API Endpoints**: 90%
- **Critical Functions**: 95%

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=app --cov-report=xml

# View coverage in terminal
pytest --cov=app --cov-report=term-missing
```

## Performance Benchmarks

### Response Time Targets
- **Fraud Detection**: < 5 seconds for 1000 transactions
- **Predictive Analytics**: < 10 seconds for comprehensive insights
- **NLP Invoice Generation**: < 2 seconds per invoice
- **API Endpoints**: < 1 second for cached responses

### Memory Usage Targets
- **Maximum Memory**: < 500MB for large datasets
- **Memory Leaks**: < 50MB increase over 10 iterations
- **Concurrent Processing**: < 1GB for 10 concurrent requests

### Load Testing Targets
- **Concurrent Users**: 50 simultaneous requests
- **Throughput**: 100 requests per minute
- **Error Rate**: < 1% under normal load

## Test Best Practices

### Writing Tests
1. **Arrange-Act-Assert**: Structure tests clearly
2. **Descriptive Names**: Use descriptive test method names
3. **Single Responsibility**: Test one thing per test method
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Edge Cases**: Include boundary conditions and error cases

### Test Data
1. **Realistic Data**: Use realistic test data
2. **Data Isolation**: Each test should be independent
3. **Cleanup**: Clean up test data after tests
4. **Fixtures**: Use fixtures for common test data

### Performance Testing
1. **Baseline Measurements**: Establish performance baselines
2. **Resource Monitoring**: Monitor CPU, memory, and I/O
3. **Load Patterns**: Test realistic load patterns
4. **Degradation Testing**: Test graceful degradation

## Continuous Integration

### GitHub Actions
```yaml
name: AI Backend Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py --ci
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Test Reports
- **JUnit XML**: `test-results.xml`
- **Coverage XML**: `coverage.xml`
- **Coverage HTML**: `htmlcov/index.html`

## Debugging Tests

### Running Individual Tests
```bash
# Run single test method
pytest tests/test_fraud_detection.py::TestFraudDetector::test_detect_duplicates

# Run with verbose output
pytest -v tests/test_fraud_detection.py

# Run with debugging
pytest --pdb tests/test_fraud_detection.py
```

### Common Issues
1. **Import Errors**: Check PYTHONPATH and module imports
2. **Async Test Issues**: Ensure proper async/await usage
3. **Mock Issues**: Verify mock setup and assertions
4. **Database Issues**: Check database connection and test data

### Logging
Enable debug logging for tests:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Adding New Tests
1. Follow existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Include docstrings explaining test purpose
4. Add fixtures to `conftest.py` if reusable
5. Update this README if adding new test categories

### Test Review Checklist
- [ ] Tests cover all code paths
- [ ] Edge cases and error conditions tested
- [ ] Performance implications considered
- [ ] Mocks used appropriately
- [ ] Test data is realistic and isolated
- [ ] Documentation updated

## Troubleshooting

### Common Test Failures
1. **Timeout Errors**: Increase timeout or optimize test performance
2. **Memory Errors**: Reduce test data size or add cleanup
3. **Flaky Tests**: Add proper waits and retries
4. **Mock Errors**: Verify mock setup and call expectations

### Getting Help
1. Check test logs for detailed error messages
2. Run tests with `-v` flag for verbose output
3. Use `--pdb` flag to debug failing tests
4. Review test documentation and examples

## Metrics and Reporting

### Test Metrics
- **Test Count**: Total number of tests
- **Coverage Percentage**: Code coverage metrics
- **Execution Time**: Test execution duration
- **Success Rate**: Percentage of passing tests

### Performance Metrics
- **Response Times**: API endpoint response times
- **Memory Usage**: Peak memory consumption
- **CPU Usage**: CPU utilization during tests
- **Throughput**: Requests processed per second

### Quality Metrics
- **Code Quality**: Linting and formatting scores
- **Complexity**: Cyclomatic complexity metrics
- **Maintainability**: Code maintainability index
- **Technical Debt**: Identified technical debt items