"""
Basic tests for the AI Backend application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Mock the database initialization to avoid requiring actual Supabase connection
with patch('app.database.init_database'):
    from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "FinEasy AI Backend"
    assert "version" in data
    assert "status" in data


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
    assert "version" in data


def test_api_documentation():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/redoc")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is generated"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "FinEasy AI Backend"


@pytest.mark.asyncio
async def test_fraud_endpoint_requires_auth():
    """Test that fraud endpoint requires authentication"""
    response = client.post("/api/v1/fraud/analyze", params={"business_id": "test"})
    assert response.status_code == 403  # Forbidden without auth


@pytest.mark.asyncio
async def test_insights_endpoint_requires_auth():
    """Test that insights endpoint requires authentication"""
    response = client.get("/api/v1/insights/test-business-id")
    assert response.status_code == 403  # Forbidden without auth