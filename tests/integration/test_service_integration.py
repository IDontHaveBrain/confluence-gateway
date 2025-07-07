"""Integration tests for service layer with real dependencies.

These tests verify that services work correctly with real adapters and minimal mocking.
Only external LLM calls are mocked according to the project's testing strategy.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService

pytestmark = [pytest.mark.integration, pytest.mark.api, pytest.mark.semantic]


class TestSearchServiceIntegration:
    """Test SearchService with real dependencies."""

    def test_keyword_search_real_confluence(
        self,
        standard_search_service: SearchService,
        real_search_term: str,
        test_space_with_content: dict | None,
    ):
        """Test keyword search with real Confluence instance using test data."""
        space_key = test_space_with_content["key"] if test_space_with_content else None

        result = standard_search_service.search_by_text(
            text=real_search_term,
            space_key=space_key,
            limit=5,
            return_enhanced_result=True,
        )

        assert result is not None
        assert result.results is not None
        assert result.statistics is not None
        assert result.statistics.execution_time_ms > 0

        if result.results.results:
            first_result = result.results.results[0]
            assert hasattr(first_result, "id")
            assert hasattr(first_result, "title")
            assert hasattr(first_result, "content_type")

            if test_space_with_content:
                # Extract space_key using the client's method
                fields = standard_search_service.client.extract_content_fields(first_result)
                assert fields.get("space_key") == space_key

    def test_semantic_search_real_vector_db(
        self,
        semantic_search_service: SearchService,
        real_search_terms: list[str],
    ):
        """Test semantic search with real vector database using real search terms."""
        if not semantic_search_service.vector_db_adapter:
            pytest.skip("Vector DB not configured")

        query = real_search_terms[0] if real_search_terms else "test documentation"

        results, took_ms = semantic_search_service.search_semantic(
            query=query,
            top_k=5,
        )

        assert isinstance(results, list)
        assert took_ms > 0

        if results:
            first_result = results[0]
            assert hasattr(first_result, "id")
            assert hasattr(first_result, "score")
            assert hasattr(first_result, "metadata")
            assert first_result.score >= 0.0 and first_result.score <= 1.0

    def test_hybrid_search_real_systems(
        self,
        semantic_search_service: SearchService,
        real_search_terms: list[str],
        test_space_with_content: dict | None,
        mocker,
    ):
        """Test hybrid search with real Confluence and vector DB using real search terms."""
        if not semantic_search_service.vector_db_adapter:
            pytest.skip("Hybrid search requires vector DB")

        mocker.patch(
            "confluence_gateway.services.search.search_config.hybrid_search_enabled",
            True,
        )

        query = real_search_terms[0] if real_search_terms else "confluence"
        space_key = test_space_with_content["key"] if test_space_with_content else None

        result = semantic_search_service.search_hybrid(
            text=query,
            space_key=space_key,
            limit=10,
            return_enhanced_result=True,
        )

        assert result is not None
        assert result.results is not None
        assert result.statistics is not None

        if result.results.results:
            assert len(result.results.results) <= 10

            if test_space_with_content:
                for search_result in result.results.results:
                    # Extract space_key using the client's method
                    fields = semantic_search_service.client.extract_content_fields(search_result)
                    assert fields.get("space_key") == space_key


class TestIndexingServiceIntegration:
    """Test IndexingService with real dependencies."""

    @pytest.mark.asyncio
    async def test_indexing_single_space(
        self,
        indexing_service: IndexingService,
        confluence_client: ConfluenceClient,
        test_space_with_content: dict | None,
    ):
        """Test indexing a single space with real data, preferring test space."""
        if not indexing_service.vector_db_adapter:
            pytest.skip("Indexing requires vector DB")

        if test_space_with_content:
            space_key = test_space_with_content["key"]
        else:
            spaces = confluence_client.list_all_spaces(limit=1)
            if not spaces:
                pytest.skip("No spaces available for testing")
            space_key = spaces[0].key

        initial_count = indexing_service.vector_db_adapter.count()

        await indexing_service.run_indexing(space_keys=[space_key])

        assert indexing_service._last_run_status in ["success", "failure"]
        assert indexing_service._last_run_end_time is not None

        if indexing_service._last_run_status == "success":
            final_count = indexing_service.vector_db_adapter.count()
            assert final_count >= initial_count

    def test_indexing_status_tracking(
        self,
        indexing_service: IndexingService,
    ):
        """Test that indexing service tracks status correctly."""
        status_info = indexing_service.status

        assert "status" in status_info
        assert status_info["status"] in ["idle", "running", "success", "failure"]

        if status_info["last_run_start_time"]:
            from datetime import datetime

            assert isinstance(status_info["last_run_start_time"], str | datetime)
        if status_info["last_run_end_time"]:
            assert isinstance(status_info["last_run_end_time"], str | datetime)


class TestGenerationServiceIntegration:
    """Test GenerationService with real dependencies."""

    @pytest.mark.asyncio
    async def test_generation_with_real_search(
        self,
        generation_service: GenerationService,
        mocker,
    ):
        """Test generation with real search results."""
        if not generation_service.config or not generation_service.config.enable:
            pytest.skip("Generation is not enabled")

        mock_llm_response = "This is a test answer based on the search results."
        mock_acompletion = mocker.patch(
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

        query = "What is Confluence?"
        answer, sources = await generation_service.generate_answer(
            query=query,
            top_k_retrieval=3,
        )

        assert answer is not None
        assert isinstance(sources, list)

        if sources:
            assert mock_acompletion.called
            assert mock_llm_response in answer
        else:
            assert "Could not find relevant information" in answer

    @pytest.mark.asyncio
    async def test_generation_context_handling(
        self,
        generation_service: GenerationService,
        real_content_samples: list[dict],
        mocker,
    ):
        """Test that generation properly handles context from search using real content."""
        if not generation_service.config or not generation_service.config.enable:
            pytest.skip("Generation is not enabled")

        if real_content_samples and len(real_content_samples) >= 2:
            from confluence_gateway.adapters.vector_db.models import (
                VectorSearchResultItem,
            )

            mock_search_results = []
            for i, content_sample in enumerate(real_content_samples[:2]):
                import re

                text_content = re.sub(r"<[^>]+>", " ", content_sample["content"])
                text_content = " ".join(text_content.split())
                text_snippet = text_content[:500]

                mock_search_results.append(
                    VectorSearchResultItem(
                        id=content_sample["id"],
                        score=0.9 - (i * 0.1),
                        metadata={
                            "title": content_sample["title"],
                            "space_key": content_sample["space_key"],
                        },
                        text=text_snippet,
                    )
                )
        else:
            from confluence_gateway.adapters.vector_db.models import (
                VectorSearchResultItem,
            )

            mock_search_results = [
                VectorSearchResultItem(
                    id="doc1",
                    score=0.9,
                    metadata={"title": "Test Doc 1"},
                    text="This is important information about Confluence.",
                ),
                VectorSearchResultItem(
                    id="doc2",
                    score=0.8,
                    metadata={"title": "Test Doc 2"},
                    text="Confluence is a collaboration tool.",
                ),
            ]

        mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=(mock_search_results, 50.0),
        )

        mock_acompletion = mocker.patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mocker.MagicMock(
                choices=[
                    mocker.MagicMock(
                        message=mocker.MagicMock(
                            content="Based on the provided context, here's what I found."
                        )
                    )
                ]
            ),
        )

        answer, sources = await generation_service.generate_answer(
            query="What can you tell me about this content?",
            top_k_retrieval=2,
        )

        call_args = mock_acompletion.call_args[1]
        prompt_content = call_args["messages"][0]["content"]

        for result in mock_search_results:
            result_words = result.text.split()[:5]
            assert any(word in prompt_content for word in result_words if len(word) > 3)


class TestServiceInteroperability:
    """Test that services work together correctly."""

    @pytest.mark.asyncio
    async def test_index_then_search_workflow(
        self,
        indexing_service: IndexingService,
        semantic_search_service: SearchService,
        confluence_client: ConfluenceClient,
        test_space_with_content: dict | None,
    ):
        """Test that indexed content can be searched using real data."""
        if not indexing_service.vector_db_adapter:
            pytest.skip("Requires vector DB for indexing")

        if not test_space_with_content:
            pytest.skip(
                "No real test data space with content available. "
                "Run 'python scripts/generate_real_data.py create' to generate test data."
            )

        space_key = test_space_with_content["key"]

        search_result = confluence_client.search(
            query="*",
            space_key=space_key,
            limit=1,
        )

        if not search_result.results:
            pytest.skip(
                f"No pages found in space {space_key} (unexpected - real test data may be corrupted)"
            )

        page_title = search_result.results[0].title
        search_keyword = page_title.split()[0] if page_title else "test"

        await indexing_service.run_indexing(space_keys=[space_key])

        time.sleep(1)

        results, took_ms = semantic_search_service.search_semantic(
            query=search_keyword,
            top_k=5,
        )

        assert isinstance(results, list)
        assert took_ms > 0

        if indexing_service._last_run_status == "success":
            assert results is not None

    def test_search_service_fallback(
        self,
        standard_search_service: SearchService,
    ):
        """Test that search service falls back gracefully when vector DB is not available."""
        service_without_vector = SearchService(
            client=standard_search_service.client,
            embedding_service=None,
            vector_db_adapter=None,
        )

        result = service_without_vector.search_by_text(
            text="test",
            limit=5,
        )

        assert result is not None
        assert result.results is not None

        with pytest.raises(Exception, match="Semantic search is not configured"):
            service_without_vector.search_semantic(query="test")
