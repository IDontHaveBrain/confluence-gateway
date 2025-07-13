import pytest
import httpx


def test_health_endpoint(api_client):
    """Test health check endpoint"""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data