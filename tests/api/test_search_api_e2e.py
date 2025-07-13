import httpx
import pytest


def test_text_search_endpoint(api_client):
    """Test GET /api/search text search"""
    response = api_client.get(
        "/api/search", params={"query": "test", "use_hybrid": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_semantic_search_endpoint(api_client):
    """Test POST /api/search/semantic endpoint"""
    response = api_client.post(
        "/api/search/semantic",
        json={"query": "test query", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_advanced_search_endpoint(api_client):
    """Test POST /api/search/advanced endpoint"""
    response = api_client.post(
        "/api/search/advanced",
        json={"query": "test", "limit": 10, "use_hybrid": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_cql_search_endpoint(api_client):
    """Test POST /api/search/cql endpoint"""
    response = api_client.post(
        "/api/search/cql", json={"cql": "text ~ test", "limit": 10}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_hybrid_search_endpoint(api_client):
    """Test GET /api/search with hybrid mode"""
    response = api_client.get(
        "/api/search", params={"query": "test", "use_hybrid": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
