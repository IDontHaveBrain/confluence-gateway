import httpx
import pytest


def test_generate_answer_endpoint(api_client):
    """Test POST /api/generate/answer endpoint"""
    response = api_client.post(
        "/api/generate/answer", json={"gen_request": {"query": "What is Confluence?"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data or "response" in data
