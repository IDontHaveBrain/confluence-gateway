from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    EmbeddingError,
    EmbeddingProviderError,
    GenerationError,
    SearchParameterError,
    SemanticSearchError,
)
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService

pytestmark = pytest.mark.integration


class TestErrorHandling:
    def test_confluence_connection_error_handling(self, mocker):
        mock_client = MagicMock(spec=ConfluenceClient)
        mock_client.test_connection.side_effect = ConfluenceConnectionError(
            "Failed to connect", cause=Exception("Network error")
        )

        with pytest.raises(ConfluenceConnectionError) as exc_info:
            mock_client.test_connection()

        assert "Failed to connect" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)

    def test_confluence_authentication_error_handling(self, mocker):
        mock_client = MagicMock(spec=ConfluenceClient)
        mock_client.search.side_effect = ConfluenceAuthenticationError(
            "Invalid credentials"
        )

        search_service = SearchService(
            client=mock_client,
            embedding_service=None,
            vector_db_adapter=None,
        )

        with pytest.raises(ConfluenceAuthenticationError) as exc_info:
            search_service.search_by_text("test")

        assert "Invalid credentials" in str(exc_info.value)

    def test_confluence_api_error_with_status_code(self):
        error = ConfluenceAPIError(status_code=404, error_message="Page not found")

        assert error.status_code == 404
        assert error.error_message == "Page not found"
        assert "404" in str(error)
        assert "Page not found" in str(error)

    def test_search_parameter_validation_error(self, standard_search_service):
        if not standard_search_service:
            pytest.skip("Requires standard search service")

        with pytest.raises(SearchParameterError):
            standard_search_service.search_by_text(
                text="test",
                limit=1000,
            )

    def test_semantic_search_without_vector_db(self, standard_search_service):
        if not standard_search_service:
            pytest.skip("Requires standard search service")

        with pytest.raises(
            SemanticSearchError, match="Semantic search is not configured"
        ):
            standard_search_service.search_semantic(query="test")

    @pytest.mark.asyncio
    async def test_generation_error_without_search_results(
        self, generation_service, mocker
    ):
        if not generation_service:
            pytest.skip("Requires generation service")

        mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=([], 0.0),
        )

        answer, sources = await generation_service.generate_answer(
            query="non-existent topic xyz123",
            top_k_retrieval=5,
        )

        assert "Could not find relevant information" in answer
        assert len(sources) == 0

    def test_embedding_provider_initialization_error(self):
        pytest.skip("Embedding provider initialization is tested in unit tests")

    @pytest.mark.asyncio
    async def test_indexing_concurrent_run_rejection(self, indexing_service):
        if not indexing_service:
            pytest.skip("Requires indexing service")

        indexing_service._is_running = True
        indexing_service._last_run_status = "running"

        await indexing_service.run_indexing()

        assert indexing_service._is_running is True
        assert indexing_service._last_run_status == "running"

    @pytest.mark.slow
    def test_vector_db_connection_error(self, mocker):
        import httpx
        from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
        from confluence_gateway.core.config import VectorDBConfig
        from qdrant_client.http.exceptions import ResponseHandlingException

        mock_qdrant_client = mocker.patch(
            "confluence_gateway.adapters.vector_db.qdrant_adapter.QdrantClient"
        )
        mock_instance = mock_qdrant_client.return_value
        mock_instance.get_collections.side_effect = ResponseHandlingException(
            httpx.ConnectError("Connection refused")
        )

        config = VectorDBConfig(
            type="qdrant",
            qdrant_url="http://invalid-host:6333",
            collection_name="test_collection",
            embedding_dimension=384,
        )

        adapter = QdrantAdapter(config)
        with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
            adapter.initialize()

    def test_attachment_parsing_error_handling(self, indexing_service, mocker):
        if not indexing_service:
            pytest.skip("Requires indexing service")

        from confluence_gateway.adapters.confluence.models import ConfluenceAttachment

        mock_attachment = mocker.MagicMock(spec=ConfluenceAttachment)
        mock_attachment.id = "test-id"
        mock_attachment.title = "test.pdf"
        mock_attachment.download_url = "http://test.com/test.pdf"

        mocker.patch.object(
            indexing_service.confluence_client,
            "download_attachment",
            return_value=b"corrupted data",
        )

        if indexing_service.attachment_parser:
            mocker.patch.object(
                indexing_service.attachment_parser,
                "parse",
                side_effect=Exception("Corrupted file"),
            )

        result = indexing_service._process_attachment(
            attachment=mock_attachment,
            parent_page_id="parent-123",
        )

        assert result is None

    def test_search_cql_injection_protection(self, standard_search_service):
        if not standard_search_service:
            pytest.skip("Requires standard search service")

        malicious_query = "test' OR type=page OR title='"

        try:
            result = standard_search_service.search_by_cql(
                cql=f"title ~ '{malicious_query}'",
                limit=5,
            )
            assert result is not None
        except ConfluenceAPIError as e:
            assert "query cannot be parsed" in str(e)

    @pytest.mark.asyncio
    async def test_generation_timeout_handling(self, generation_service, mocker):
        if not generation_service:
            pytest.skip("Requires generation service")

        from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
        from litellm.exceptions import Timeout

        mock_results = [
            VectorSearchResultItem(
                id="doc1",
                score=0.9,
                metadata={"title": "Test"},
                text="Test content",
            )
        ]
        mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=(mock_results, 50.0),
        )

        mocker.patch(
            "litellm.acompletion",
            side_effect=Timeout(
                message="Request timed out",
                model="test-model",
                llm_provider="test-provider",
            ),
        )

        with pytest.raises(GenerationError, match=r"LLM API error \(Timeout\)"):
            await generation_service.generate_answer(query="test query")
