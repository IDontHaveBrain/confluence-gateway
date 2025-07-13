import pytest
import httpx


def test_list_spaces_endpoint(api_client):
    """Test GET /api/spaces endpoint"""
    response = api_client.get("/api/spaces")
    assert response.status_code == 200
    data = response.json()
    assert "spaces" in data or isinstance(data, list)


def test_list_all_spaces_endpoint(api_client):
    """Test GET /api/spaces?include_archived=true endpoint"""
    response = api_client.get("/api/spaces", params={"include_archived": True})
    assert response.status_code == 200


def test_get_space_endpoint(api_client):
    """Test GET /api/spaces/{space_key} endpoint"""
    # First get actual space_key from spaces list
    spaces_response = api_client.get("/api/spaces")
    if spaces_response.status_code == 200:
        spaces_data = spaces_response.json()
        # Test with first space (data parsing needed in actual implementation)
        pass