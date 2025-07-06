"""End-to-end workflow tests for Confluence Gateway."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from confluence_gateway.cli.main import app as cli_app
from fastapi.testclient import TestClient
from typer.testing import CliRunner

pytestmark = [pytest.mark.integration, pytest.mark.api, pytest.mark.semantic]


class TestE2EWorkflows:
    """Test complete end-to-end workflows."""

    def test_full_indexing_search_generation_workflow_api(
        self,
        test_app_client: TestClient,
        is_generation_enabled: bool,
        is_semantic_search_possible: bool,
        test_space_with_content: dict | None,
        real_search_terms: list[str],
        mocker,
    ):
        """Test the complete workflow: Index → Search → Generate via API using real test data."""
        if not is_semantic_search_possible:
            pytest.skip("Requires semantic search capabilities")
        if not is_generation_enabled:
            pytest.skip("Generation feature is disabled")

        if test_space_with_content:
            space_key = test_space_with_content["key"]

            response = test_app_client.post(
                "/api/indexing/trigger", json={"space_keys": [space_key], "force": True}
            )
        else:
            response = test_app_client.post("/api/indexing/trigger", json={})

        if response.status_code == 409:
            for _ in range(30):
                status_response = test_app_client.get("/api/indexing/status")
                status_data = status_response.json()
                if status_data["status"] != "running":
                    break
                time.sleep(1)
        else:
            assert response.status_code in [200, 202]
            time.sleep(2)

        search_term = real_search_terms[0] if real_search_terms else "documentation"
        search_payload = {"query": search_term, "top_k": 5}
        search_response = test_app_client.post(
            "/api/search/semantic", json=search_payload
        )
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert "results" in search_data

        mock_llm_response = (
            f"Based on the search results, I found information about {search_term}."
        )
        mocker.patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mocker.MagicMock(
                choices=[
                    mocker.MagicMock(
                        message=mocker.MagicMock(content=mock_llm_response)
                    )
                ]
            ),
        )

        generation_payload = {
            "query": f"What can you tell me about {search_term}?",
            "top_k_retrieval": 3,
        }
        generation_response = test_app_client.post(
            "/api/generate/answer", json=generation_payload
        )
        assert generation_response.status_code == 200
        generation_data = generation_response.json()
        assert "answer" in generation_data
        assert "sources" in generation_data

        if search_data["results"]:
            assert mock_llm_response in generation_data["answer"]
            assert len(generation_data["sources"]) > 0
        else:
            assert (
                "Could not find" in generation_data["answer"]
                or mock_llm_response in generation_data["answer"]
            )

    def test_full_workflow_cli(
        self,
        runner: CliRunner,
        is_generation_enabled: bool,
        is_semantic_search_possible: bool,
        real_search_terms: list[str],
        mocker,
        confluence_config,
    ):
        """Test the complete workflow: Index → Search → Generate via CLI using real search terms."""
        if not is_semantic_search_possible:
            pytest.skip("Requires semantic search capabilities")
        if not is_generation_enabled:
            pytest.skip("Generation feature is disabled")

        with patch(
            "confluence_gateway.cli.dependencies.confluence_config", confluence_config
        ):
            status_result = runner.invoke(cli_app, ["index", "status"])
            assert status_result.exit_code == 0

            search_term = real_search_terms[0] if real_search_terms else "confluence"
            search_result = runner.invoke(
                cli_app, ["search", "semantic", search_term, "--top-k", "3"]
            )
            assert search_result.exit_code == 0
            assert "Semantic Search Results" in search_result.stdout

            mock_llm_response = f"Based on the available information about {search_term}, here's what I found."
            mocker.patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mocker.MagicMock(
                    choices=[
                        mocker.MagicMock(
                            message=mocker.MagicMock(content=mock_llm_response)
                        )
                    ]
                ),
            )

            generate_result = runner.invoke(
                cli_app,
                [
                    "generate",
                    "answer",
                    f"What can you tell me about {search_term}?",
                    "--top-k",
                    "3",
                ],
            )
            assert generate_result.exit_code == 0
            assert "Generated Answer" in generate_result.stdout
            assert "answer" in generate_result.stdout.lower()

    def test_hybrid_search_workflow(
        self,
        test_app_client: TestClient,
        is_semantic_search_possible: bool,
        is_real_config_available: bool,
        real_search_terms: list[str],
        test_space_with_content: dict | None,
        mocker,
    ):
        """Test hybrid search combining keyword and semantic search with real data."""
        if not is_semantic_search_possible:
            pytest.skip("Hybrid search requires semantic search capabilities")

        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration")

        mocker.patch(
            "confluence_gateway.services.search.search_config.hybrid_search_enabled",
            True,
        )

        search_term = real_search_terms[0] if real_search_terms else "documentation"

        if test_space_with_content:
            space_key = test_space_with_content["key"]
            query_params = (
                f"query={search_term}&use_hybrid=true&space_key={space_key}&limit=10"
            )
        else:
            query_params = f"query={search_term}&use_hybrid=true&limit=10"

        response = test_app_client.get(f"/api/search?{query_params}")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

        if len(data["results"]) > 1:
            for result in data["results"]:
                assert "id" in result
                assert "title" in result
                assert "type" in result
                assert "space_key" in result

                if test_space_with_content:
                    assert result["space_key"] == space_key

    def test_attachment_indexing_workflow(
        self,
        test_app_client: TestClient,
        is_semantic_search_possible: bool,
        test_space_with_attachments: dict | None,
        mocker,
    ):
        """Test indexing and searching attachments using dummy data."""
        if not is_semantic_search_possible:
            pytest.skip("Requires semantic search capabilities")

        if not test_space_with_attachments:
            pytest.skip(
                "No dummy data space with attachments available. "
                "Run 'python scripts/generate_dummy_data.py create' to generate test data with attachments."
            )

        space_key = test_space_with_attachments["key"]
        page_id = test_space_with_attachments["sample_page_id"]

        mock_indexing_config = mocker.patch(
            "confluence_gateway.api.dependencies.get_indexing_config"
        )
        mock_indexing_config.return_value = mocker.MagicMock(
            process_attachments=True,
            max_file_size_mb=50,
            supported_formats=["pdf", "docx", "txt", "md"],
        )

        response = test_app_client.post(
            "/api/indexing/trigger", json={"space_keys": [space_key], "force": True}
        )

        assert response.status_code in [200, 202]

        if response.status_code == 202:
            time.sleep(3)

        search_response = test_app_client.get(
            f"/api/search?query=space={space_key}&content_type=attachment&limit=10"
        )
        assert search_response.status_code == 200
        data = search_response.json()

        assert "results" in data
        assert "statistics" in data

        if data["results"]:
            for result in data["results"]:
                if result["type"] == "attachment":
                    assert "space_key" in result
                    assert "url" in result

    def test_error_handling_workflow(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
        mocker,
    ):
        """Test error handling throughout the workflow."""
        response = test_app_client.get("/api/search?query=a")

        if not is_real_config_available:
            assert response.status_code == 503
        else:
            assert response.status_code == 422

        mocker.patch(
            "confluence_gateway.api.dependencies.generation_config",
            None,
        )
        response = test_app_client.post("/api/generate/answer", json={"query": "test"})
        assert response.status_code == 501

    def test_space_filtering_workflow(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
        test_space_with_content: dict | None,
        real_search_terms: list[str],
    ):
        """Test searching within specific spaces using real test data."""
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration")

        search_term = real_search_terms[0] if real_search_terms else "test"

        if test_space_with_content:
            space_key = test_space_with_content["key"]
        else:
            response = test_app_client.get(f"/api/search?query={search_term}&limit=5")
            assert response.status_code == 200
            data = response.json()

            if not data["results"]:
                pytest.skip("No search results found to extract space key")

            space_key = data["results"][0]["space_key"]

        space_response = test_app_client.get(
            f"/api/search?query={search_term}&space_key={space_key}&limit=5"
        )
        assert space_response.status_code == 200
        space_data = space_response.json()

        for result in space_data["results"]:
            assert result["space_key"] == space_key

    def test_concurrent_operations(
        self,
        test_app_client: TestClient,
        is_real_config_available: bool,
    ):
        """Test concurrent search operations."""
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration")

        import concurrent.futures

        def search_request(query: str):
            return test_app_client.get(f"/api/search?query={query}&limit=2")

        queries = ["test", "documentation", "confluence", "api", "search"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_request, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for response in results:
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
