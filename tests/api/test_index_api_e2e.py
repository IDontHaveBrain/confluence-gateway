import httpx
import pytest


def test_trigger_indexing_endpoint(api_client):
    """Test POST /api/index/trigger endpoint"""
    response = api_client.post("/api/index/trigger", json={"request": {}})
    assert response.status_code in [200, 202]  # Success or accepted


def test_indexing_status_endpoint(api_client):
    """Test GET /api/index/status endpoint"""
    response = api_client.get("/api/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
