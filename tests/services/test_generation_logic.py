from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.services.generation import GenerationError, GenerationService
from confluence_gateway.services.search import SearchService
from litellm.exceptions import APIConnectionError, Timeout


@pytest.mark.integration
@pytest.mark.semantic
class TestGenerationServiceLogic:
    @pytest.fixture
    def mock_search_results(self) -> list[VectorSearchResultItem]:
        return [
            VectorSearchResultItem(
                id="doc1_chunk0",
                score=0.9,
                metadata={"title": "Doc 1"},
                text="This is the first context document.",
            ),
            VectorSearchResultItem(
                id="doc2_chunk0",
                score=0.8,
                metadata={"title": "Doc 2"},
                text="Second piece of context information.",
            ),
        ]

    @pytest.fixture
    def mock_llm_response(self) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = "This is the generated answer."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_generate_answer_success(
        self,
        generation_service: GenerationService,
        mocker,
        mock_search_results,
        mock_llm_response,
    ):
        query = "What is the context about?"
        mock_search_semantic = mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=(mock_search_results, 50.0),
        )
        mock_acompletion = mocker.patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        )
        answer, retrieved_docs = await generation_service.generate_answer(
            query=query, top_k_retrieval=2
        )
        mock_search_semantic.assert_called_once_with(query=query, top_k=2, filters=None)
        mock_acompletion.assert_awaited_once()
        call_args = mock_acompletion.call_args[1]
        assert call_args["model"] == generation_service.config.model_name
        assert call_args["max_tokens"] == generation_service.config.max_output_tokens
        assert call_args["temperature"] == generation_service.config.temperature
        expected_context = "This is the first context document.\n\n---\n\nSecond piece of context information."
        expected_prompt_text = generation_service.config.prompt_template.format(
            context=expected_context, query=query
        )
        assert call_args["messages"] == [
            {"role": "user", "content": expected_prompt_text}
        ]
        assert answer == "This is the generated answer."
        assert retrieved_docs == mock_search_results

    @pytest.mark.asyncio
    async def test_generate_answer_no_context_found(
        self, generation_service: GenerationService, mocker
    ):
        query = "Query with no context"
        mock_search_semantic = mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=([], 10.0),
        )
        mock_acompletion = mocker.patch("litellm.acompletion", new_callable=AsyncMock)
        answer, retrieved_docs = await generation_service.generate_answer(query=query)
        mock_search_semantic.assert_called_once_with(query=query, top_k=5, filters=None)
        mock_acompletion.assert_not_awaited()
        assert "Could not find relevant information" in answer
        assert retrieved_docs == []

    @pytest.mark.asyncio
    async def test_generate_answer_llm_api_error(
        self, generation_service: GenerationService, mocker, mock_search_results
    ):
        query = "Query causing LLM error"
        mock_search_semantic = mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=(mock_search_results, 50.0),
        )
        mock_acompletion = mocker.patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Timeout(
                message="Request timed out",
                model="test-model",
                llm_provider="test-provider",
            ),
        )
        with pytest.raises(
            GenerationError,
            match=r"LLM API error \(Timeout\): litellm\.Timeout: Request timed out",
        ):
            await generation_service.generate_answer(query=query)
        mock_search_semantic.assert_called_once()
        mock_acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_answer_context_truncation(
        self, generation_service: GenerationService, mocker, mock_llm_response
    ):
        query = "Test truncation"
        long_text = "word " * 1000
        mock_search_results_long = [
            VectorSearchResultItem(id="long1", score=0.9, metadata={}, text=long_text),
            VectorSearchResultItem(id="long2", score=0.8, metadata={}, text=long_text),
        ]
        mocker.patch.object(
            generation_service.search_service,
            "search_semantic",
            return_value=(mock_search_results_long, 50.0),
        )
        mock_acompletion = mocker.patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        )
        original_max_tokens = generation_service.config.max_context_tokens
        generation_service.config.max_context_tokens = 50
        if not generation_service.tokenizer:
            pytest.skip("Skipping truncation test: TikToken tokenizer not available.")
        try:
            await generation_service.generate_answer(query=query)
        finally:
            generation_service.config.max_context_tokens = original_max_tokens
        mock_acompletion.assert_awaited_once()
        call_args = mock_acompletion.call_args[1]
        prompt_content = call_args["messages"][0]["content"]
        context_start_marker = "Context:\n"
        context_end_marker = "\n\nQuestion:"
        context_part = prompt_content[
            len(context_start_marker) : prompt_content.find(context_end_marker)
        ]
        token_count = generation_service._count_tokens(context_part)
        assert token_count <= generation_service.config.max_context_tokens + 10
