import pytest
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.api.app import app
from confluence_gateway.api.dependencies import get_embedding_provider_dependency

from tests.conftest import REAL_CONFIG_SKIP_REASON, SEMANTIC_SEARCH_SKIP_REASON


@pytest.fixture(scope="class", autouse=True)
def override_embedding_provider_for_test(embedding_provider, embedding_config, request):
    """
    Overrides the embedding provider dependency for semantic search tests.
    Uses the centralized embedding_provider fixture instead of creating a new instance.
    """
    original_override = app.dependency_overrides.get(get_embedding_provider_dependency)
    override_applied = False

    if embedding_config is None or embedding_config.provider == "none":
        if embedding_provider:
            print(
                "\nINFO: Using pytest embedding_provider fixture for FastAPI dependency override"
            )

            def get_test_embedding_provider():
                return embedding_provider

            app.dependency_overrides[get_embedding_provider_dependency] = (
                get_test_embedding_provider
            )
            override_applied = True
            print(
                "INFO: Embedding Provider dependency overridden with pytest fixture instance."
            )
        else:
            print(
                "\nINFO: No embedding provider available from fixture, skipping override"
            )
    else:
        print("\nINFO: Using globally configured Embedding Provider, skipping override")

    yield

    if override_applied:
        if original_override:
            app.dependency_overrides[get_embedding_provider_dependency] = (
                original_override
            )
        else:
            if get_embedding_provider_dependency in app.dependency_overrides:
                del app.dependency_overrides[get_embedding_provider_dependency]
        print("INFO: Restored original Embedding Provider dependency.")


@pytest.fixture(scope="class", autouse=True)
def override_vector_db_for_test(vector_db_adapter, vector_db_config, request):
    """
    Overrides the vector DB dependency for semantic search tests.
    Uses the centralized vector_db_adapter fixture instead of creating a new instance.
    """
    original_override = app.dependency_overrides.get(get_vector_db_adapter)
    override_applied = False

    if vector_db_config is None or vector_db_config.type == "none":
        if vector_db_adapter:
            print(
                "\nINFO: Using pytest vector_db_adapter fixture for FastAPI dependency override"
            )

            def get_test_vector_db_adapter():
                return vector_db_adapter

            app.dependency_overrides[get_vector_db_adapter] = get_test_vector_db_adapter
            override_applied = True
            print("INFO: Vector DB dependency overridden with pytest fixture instance.")
        else:
            print(
                "\nINFO: No vector DB adapter available from fixture, skipping override"
            )
    else:
        print("\nINFO: Using globally configured Vector DB, skipping override")

    yield

    if override_applied:
        if original_override:
            app.dependency_overrides[get_vector_db_adapter] = original_override
        else:
            if get_vector_db_adapter in app.dependency_overrides:
                del app.dependency_overrides[get_vector_db_adapter]
        print("INFO: Restored original Vector DB dependency.")


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestBasicSearchFlow:
    def test_search_api_endpoint_returns_results(
        self, test_app_client, real_search_term
    ):
        response = test_app_client.get(f"/api/search?query={real_search_term}")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

    def test_search_result_format(self, test_app_client, real_search_term):
        response = test_app_client.get(f"/api/search?query={real_search_term}")
        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert "start" in data
        assert "limit" in data
        assert "took_ms" in data
        assert "page_count" in data
        assert "current_page" in data
        assert "has_more" in data

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
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestCQLSearchFlow:
    def test_cql_search_api_endpoint(
        self, test_app_client, real_search_term, confluence_client
    ):
        try:
            spaces_result = confluence_client.atlassian_api.get_all_spaces(limit=1)
            space_key = (
                spaces_result["results"][0]["key"]
                if spaces_result.get("results")
                else None
            )
        except Exception:
            space_key = None

        cql = f'text ~ "{real_search_term}"'
        if space_key:
            cql += f' AND space = "{space_key}"'

        response = test_app_client.post(
            "/api/search/cql", json={"cql": cql, "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestAdvancedSearchFlow:
    def test_advanced_search_api_endpoint(self, test_app_client, real_search_term):
        response = test_app_client.post(
            "/api/search/advanced",
            json={
                "query": real_search_term,
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

        if len(data["results"]) > 1:
            titles = [item["title"] for item in data["results"]]
            sorted_titles = sorted(titles)
            assert titles == sorted_titles

    def test_advanced_search_with_filters(self, test_app_client, real_search_term):
        response = test_app_client.post(
            "/api/search/advanced",
            json={"query": real_search_term, "content_type": "page", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        if data["total"] > 0:
            for result in data["results"]:
                assert result["type"] == "page"


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestPaginationFlow:
    def test_pagination_links_and_navigation(self, test_app_client, real_search_term):
        response = test_app_client.get(f"/api/search?query={real_search_term}&limit=2")
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

        response = test_app_client.get(f"/{next_path}")
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
    not pytest.lazy_fixture("is_semantic_search_possible"),
    reason=SEMANTIC_SEARCH_SKIP_REASON,
)
@pytest.mark.integration
class TestSemanticSearchFlow:
    def test_semantic_search_api_endpoint(self, test_app_client, real_search_term):
        payload = {"query": real_search_term, "top_k": 5}
        response = test_app_client.post("/api/search/semantic", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "took_ms" in data
        assert "query" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["took_ms"], (float, int))
        assert data["took_ms"] >= 0
        assert data["query"] == real_search_term

        if data["results"]:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "score" in first_result
            assert isinstance(first_result["score"], (float, int))
            assert "metadata" in first_result
            assert isinstance(first_result["metadata"], dict)

    def test_semantic_search_invalid_input(self, test_app_client):
        payload = {"query": "", "top_k": 5}
        response = test_app_client.post("/api/search/semantic", json=payload)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert any("query" in item.get("loc", []) for item in data["detail"]), (
            "Error detail should mention the 'query' field"
        )


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestErrorPropagation:
    def test_invalid_parameter_error(self, test_app_client):
        response = test_app_client.get("/api/search?query=a")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_invalid_cql_error(self, test_app_client):
        response = test_app_client.post(
            "/api/search/cql", json={"cql": "&&&invalidcql!!!"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_content_type_validation(self, test_app_client):
        response = test_app_client.post(
            "/api/search/advanced",
            json={"query": "test", "content_type": "invalid_type"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"), reason=REAL_CONFIG_SKIP_REASON
)
@pytest.mark.integration
class TestDataConsistency:
    def test_cross_endpoint_result_consistency(self, test_app_client, real_search_term):
        basic_response = test_app_client.get(
            f"/api/search?query={real_search_term}&limit=5"
        )
        advanced_response = test_app_client.post(
            "/api/search/advanced", json={"query": real_search_term, "limit": 5}
        )

        assert basic_response.status_code == 200
        assert advanced_response.status_code == 200

        basic_data = basic_response.json()
        advanced_data = advanced_response.json()

        assert basic_data["total"] == advanced_data["total"]

        assert len(basic_data["results"]) == len(advanced_data["results"])
