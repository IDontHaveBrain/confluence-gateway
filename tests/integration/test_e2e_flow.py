import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.api.app import app
from confluence_gateway.core.config import load_configurations
from fastapi.testclient import TestClient

client = TestClient(app)

_confluence_config, _, _, _ = load_configurations()
real_config_available = _confluence_config is not None


@pytest.fixture
def confluence_client():
    config, _, _, _ = load_configurations()
    if not config:
        pytest.skip(
            "Confluence configuration not available - skipping integration tests"
        )
    return ConfluenceClient(config=config)


@pytest.fixture
def search_term(confluence_client):
    try:
        common_terms = ["the", "and", "is", "in", "to"]

        for term in common_terms:
            result = confluence_client.search(query=term, limit=1)
            if result.total_size > 0:
                return term

        spaces_result = confluence_client.atlassian_api.get_all_spaces(limit=1)
        if spaces_result and "results" in spaces_result and spaces_result["results"]:
            space_name = spaces_result["results"][0].get("name")
            if space_name:
                return space_name

        pytest.skip("Could not find any search terms that return results")
    except Exception:
        pytest.skip("Error occurred trying to find valid search term")


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestBasicSearchFlow:
    def test_search_api_endpoint_returns_results(self, search_term):
        response = client.get(f"/api/search?query={search_term}")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

    def test_search_result_format(self, search_term):
        response = client.get(f"/api/search?query={search_term}")
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert "total" in data
        assert "start" in data
        assert "limit" in data
        assert "took_ms" in data
        assert "page_count" in data
        assert "current_page" in data
        assert "has_more" in data

        # Check result item structure
        if data["total"] > 0 and len(data["results"]) > 0:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "title" in first_result
            assert "type" in first_result
            assert "space_key" in first_result
            assert "space_name" in first_result
            assert "url" in first_result
            assert "last_modified" in first_result


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestCQLSearchFlow:
    def test_cql_search_api_endpoint(self, search_term, confluence_client):
        try:
            spaces_result = confluence_client.atlassian_api.get_all_spaces(limit=1)
            space_key = (
                spaces_result["results"][0]["key"]
                if spaces_result.get("results")
                else None
            )
        except Exception:
            space_key = None

        cql = f'text ~ "{search_term}"'
        if space_key:
            cql += f' AND space = "{space_key}"'

        response = client.post("/api/search/cql", json={"cql": cql, "limit": 10})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestAdvancedSearchFlow:
    def test_advanced_search_api_endpoint(self, search_term):
        response = client.post(
            "/api/search/advanced",
            json={
                "query": search_term,
                "limit": 10,
                "sort_by": ["title"],
                "sort_direction": ["asc"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

        # If we have multiple results, verify sorting works
        if len(data["results"]) > 1:
            titles = [item["title"] for item in data["results"]]
            sorted_titles = sorted(titles)
            assert titles == sorted_titles

    def test_advanced_search_with_filters(self, search_term):
        response = client.post(
            "/api/search/advanced",
            json={"query": search_term, "content_type": "page", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        if data["total"] > 0:
            for result in data["results"]:
                assert result["type"] == "page"


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestPaginationFlow:
    def test_pagination_links_and_navigation(self, search_term):
        response = client.get(f"/api/search?query={search_term}&limit=2")
        assert response.status_code == 200
        first_page = response.json()

        if first_page["total"] <= 2:
            pytest.skip("Not enough results to test pagination")

        assert first_page["has_more"]
        assert "links" in first_page
        assert "next" in first_page["links"]

        next_link = first_page["links"]["next"]
        next_path = (
            next_link.split("://")[-1].split("/", 1)[-1]
            if "://" in next_link
            else next_link
        )

        response = client.get(f"/{next_path}")
        assert response.status_code == 200
        second_page = response.json()

        assert second_page["start"] > first_page["start"]
        assert "links" in second_page
        assert "previous" in second_page["links"]

        first_page_ids = [item["id"] for item in first_page["results"]]
        second_page_ids = [item["id"] for item in second_page["results"]]

        if first_page_ids == second_page_ids:
            assert second_page["current_page"] > first_page["current_page"]
        else:
            assert not any(id in first_page_ids for id in second_page_ids)


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestErrorPropagation:
    def test_invalid_parameter_error(self):
        response = client.get("/api/search?query=a")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_invalid_cql_error(self):
        response = client.post("/api/search/cql", json={"cql": "&&&invalidcql!!!"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Validation errors have a different structure in FastAPI default responses
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_content_type_validation(self):
        response = client.post(
            "/api/search/advanced",
            json={"query": "test", "content_type": "invalid_type"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Validation errors have a different structure in FastAPI default responses
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestDataConsistency:
    def test_cross_endpoint_result_consistency(self, search_term):
        # Get results from both endpoints with the same query
        basic_response = client.get(f"/api/search?query={search_term}&limit=5")
        advanced_response = client.post(
            "/api/search/advanced", json={"query": search_term, "limit": 5}
        )

        assert basic_response.status_code == 200
        assert advanced_response.status_code == 200

        basic_data = basic_response.json()
        advanced_data = advanced_response.json()

        # Total should be the same
        assert basic_data["total"] == advanced_data["total"]

        # Result counts should match
        # (might be less than limit if not enough results)
        assert len(basic_data["results"]) == len(advanced_data["results"])
