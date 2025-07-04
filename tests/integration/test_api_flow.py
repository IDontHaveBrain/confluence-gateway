import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.api]


class TestApiFlows:
    def test_health_check(self, test_app_client: TestClient):
        response = test_app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "confluence_connection" in data

    def test_search_text_basic(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
        real_search_term: str,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")

        response = test_app_client.get(f"/api/search?query={real_search_term}")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "total" in data
        assert "start" in data
        assert "limit" in data
        assert "took_ms" in data
        assert "page_count" in data
        assert "current_page" in data
        assert "has_more" in data
        assert "links" in data

        if data["total"] > 0 and data["results"]:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "title" in first_result
            assert "type" in first_result
            assert "space_key" in first_result
            assert "space_name" in first_result
            assert "url" in first_result
            assert "last_modified" in first_result

    def test_search_text_with_limit(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
        real_search_term: str,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")

        limit = 1
        response = test_app_client.get(
            f"/api/search?query={real_search_term}&limit={limit}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= limit
        assert data["limit"] == limit

    def test_search_semantic(
        self,
        test_app_client: TestClient,
        is_semantic_search_possible: bool,
    ):
        if not is_semantic_search_possible:
            pytest.skip(
                "Requires semantic search configuration (Confluence, Embedding, VectorDB)."
            )

        search_payload = {"query": "apples", "top_k": 2}
        response = test_app_client.post("/api/search/semantic", json=search_payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "took_ms" in data
        assert "query" in data
        assert data["query"] == search_payload["query"]

        if data["results"]:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "score" in first_result
            assert isinstance(first_result["score"], float)
            assert "metadata" in first_result
            assert isinstance(first_result["metadata"], dict)
            assert "source" in first_result["metadata"]
            assert "text" in first_result

    def test_search_cql(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")

        cql_payload = {"cql": "type=page", "limit": 1}
        response = test_app_client.post("/api/search/cql", json=cql_payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "total" in data
        assert data["limit"] == cql_payload["limit"]
        assert len(data["results"]) <= cql_payload["limit"]

    def test_search_advanced(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
        real_search_term: str,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")

        advanced_payload = {"query": real_search_term, "limit": 1}
        response = test_app_client.post("/api/search/advanced", json=advanced_payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "total" in data
        assert data["limit"] == advanced_payload["limit"]

    def test_search_hybrid(
        self,
        test_app_client: TestClient,
        is_semantic_search_possible: bool,
        real_search_term: str,
        search_config,
        mocker,
    ):
        if not is_semantic_search_possible:
            pytest.skip("Hybrid search requires semantic search capabilities.")

        mocker.patch(
            "confluence_gateway.services.search.search_config.hybrid_search_enabled",
            True,
        )

        response = test_app_client.get(
            f"/api/search?query={real_search_term}&use_hybrid=true&limit=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "total" in data

    def test_indexing_trigger_and_status(
        self,
        test_app_client: TestClient,
        is_semantic_search_possible: bool,
    ):
        if not is_semantic_search_possible:
            pytest.skip(
                "Indexing requires semantic search configuration (Confluence, Embedding, VectorDB)."
            )

        response_status1 = test_app_client.get("/api/indexing/status")
        assert response_status1.status_code == 200
        status_data1 = response_status1.json()
        assert "status" in status_data1
        initial_status = status_data1["status"]

        trigger_payload = {}
        response_trigger = test_app_client.post(
            "/api/indexing/trigger", json=trigger_payload
        )

        if initial_status == "running":
            assert response_trigger.status_code == 409
            data = response_trigger.json()
            assert data.get("code") == 409
            assert "already running" in data.get("message", "").lower()
        else:
            assert response_trigger.status_code == 202
            data = response_trigger.json()
            assert "accepted" in data.get("message", "").lower()

            import time

            time.sleep(0.5)
            response_status2 = test_app_client.get("/api/indexing/status")
            assert response_status2.status_code == 200
            status_data2 = response_status2.json()
            assert status_data2["status"] in ["running", "idle", "success"]

    def test_generate_answer(
        self,
        test_app_client: TestClient,
        is_generation_enabled: bool,
        is_semantic_search_possible: bool,
    ):
        if not is_generation_enabled:
            pytest.skip("Generation feature is disabled.")
        if not is_semantic_search_possible:
            pytest.skip("Generation requires semantic search capabilities.")

        generation_payload = {
            "query": "What fruits are mentioned?",
            "top_k_retrieval": 3,
        }
        response = test_app_client.post("/api/generate/answer", json=generation_payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "sources" in data
        assert isinstance(data["sources"], list)

        if data["sources"]:
            first_source = data["sources"][0]
            assert "id" in first_source
            assert "score" in first_source
            assert isinstance(first_source["score"], float)
            assert "title" in first_source
            assert "url" in first_source
            assert "space_key" in first_source

    def test_search_invalid_input(self, test_app_client: TestClient):
        response = test_app_client.get("/api/search?query=q")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert any("query" in item.get("loc", []) for item in data["detail"])
        assert any(
            "at least 2 characters" in item.get("msg", "") for item in data["detail"]
        )
