import typing
from unittest.mock import MagicMock

import litellm
import pytest
from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.embedding.litellm import LiteLLMProvider
from confluence_gateway.adapters.embedding.sentence_transformer import (
    SentenceTransformerProvider,
)
from confluence_gateway.core.config import EmbeddingConfig
from confluence_gateway.core.exceptions import EmbeddingProviderError
from pytest_mock import MockerFixture

from tests.conftest import MockedProviderFixture


@pytest.mark.integration
@pytest.mark.semantic
class TestSentenceTransformerProviderIntegration:
    @pytest.fixture(scope="class")
    def st_provider(self, embedding_provider) -> SentenceTransformerProvider:
        if not isinstance(embedding_provider, SentenceTransformerProvider):
            pytest.skip("Fixture is not a SentenceTransformerProvider")
        return embedding_provider

    def test_st_provider_initialization(self, st_provider: SentenceTransformerProvider):
        assert st_provider is not None
        assert st_provider.model is not None
        assert st_provider.device is not None

    def test_st_provider_get_dimension(self, st_provider: SentenceTransformerProvider):
        dimension = st_provider.get_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0
        assert dimension == st_provider.config.dimension

    def test_st_provider_embed_text(self, st_provider: SentenceTransformerProvider):
        text = "This is a test sentence."
        embedding = st_provider.embed_text(text)
        assert isinstance(embedding, list)
        assert len(embedding) == st_provider.get_dimension()
        assert all(isinstance(x, float) for x in embedding)

    def test_st_provider_embed_texts(self, st_provider: SentenceTransformerProvider):
        texts = ["First sentence.", "Second sentence, slightly longer."]
        embeddings = st_provider.embed_texts(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == st_provider.get_dimension() for emb in embeddings)
        assert all(isinstance(x, float) for emb in embeddings for x in emb)

    def test_st_provider_embed_empty_text(
        self, st_provider: SentenceTransformerProvider
    ):
        assert st_provider.embed_text("") == []
        assert st_provider.embed_text(None) == []

    def test_st_provider_embed_empty_texts(
        self, st_provider: SentenceTransformerProvider
    ):
        assert st_provider.embed_texts([]) == []
        assert st_provider.embed_texts(["", None, "valid"]) == [
            st_provider.embed_text("valid")
        ]
        assert st_provider.embed_texts(["", None]) == []

    def test_st_provider_close(self, st_provider: SentenceTransformerProvider):
        temp_provider = SentenceTransformerProvider(st_provider.config)
        temp_provider.initialize()
        assert temp_provider.model is not None
        temp_provider.close()
        assert temp_provider.model is None


@pytest.fixture
def litellm_provider_and_mock(mocked_litellm_provider) -> MockedProviderFixture:
    if not mocked_litellm_provider:
        pytest.skip("Mocked LiteLLM provider fixture not available.")
    return mocked_litellm_provider


@pytest.mark.unit
class TestLiteLLMProviderMocked:
    def test_litellm_provider_initialization(
        self, litellm_provider_and_mock: MockedProviderFixture, mocker: MockerFixture
    ):
        litellm_provider, mock_embedding_call = litellm_provider_and_mock
        dummy_embedding = [0.1] * litellm_provider.get_dimension()
        mock_response_data = [{"embedding": dummy_embedding}]
        mock_response_obj = MagicMock()
        mock_response_obj.data = mock_response_data
        mock_embedding_call.return_value = mock_response_obj

        mocker.patch.object(
            litellm_provider,
            "_validate_embedding_response",
            return_value=mock_response_data,
        )
        mocker.patch.object(
            litellm_provider,
            "_extract_embedding_from_item",
            return_value=dummy_embedding,
        )

        litellm_provider.initialize()

        mock_embedding_call.assert_called_once()
        call_args, call_kwargs = mock_embedding_call.call_args
        assert call_kwargs.get("model") == litellm_provider.config.model_name
        assert call_kwargs.get("input") == ["validate provider initialization"]

    def test_litellm_provider_get_dimension(
        self, litellm_provider_and_mock: MockedProviderFixture
    ):
        litellm_provider, _ = litellm_provider_and_mock
        dimension = litellm_provider.get_dimension()
        assert isinstance(dimension, int)
        assert dimension == litellm_provider.config.dimension

    def test_litellm_provider_embed_text(
        self, litellm_provider_and_mock: MockedProviderFixture, mocker: MockerFixture
    ):
        litellm_provider, mock_embedding_call = litellm_provider_and_mock
        dummy_embedding = [0.1] * litellm_provider.get_dimension()
        mock_response_data = [{"embedding": dummy_embedding}]
        mock_response_obj = MagicMock()
        mock_response_obj.data = mock_response_data
        mock_embedding_call.return_value = mock_response_obj
        mocker.patch.object(
            litellm_provider,
            "_validate_embedding_response",
            return_value=mock_response_data,
        )
        mocker.patch.object(
            litellm_provider,
            "_extract_embedding_from_item",
            return_value=dummy_embedding,
        )

        text = "Embed this text via mock."
        embedding = litellm_provider.embed_text(text)

        mock_embedding_call.assert_called_once()
        call_args, call_kwargs = mock_embedding_call.call_args
        assert call_kwargs.get("model") == litellm_provider.config.model_name
        assert call_kwargs.get("input") == [text]
        assert call_kwargs.get("api_key") == litellm_provider.config.litellm_api_key
        assert call_kwargs.get("api_base") == (
            str(litellm_provider.config.litellm_api_base)
            if litellm_provider.config.litellm_api_base
            else None
        )

        assert embedding == dummy_embedding

    def test_litellm_provider_embed_texts(
        self, litellm_provider_and_mock: MockedProviderFixture, mocker: MockerFixture
    ):
        litellm_provider, mock_embedding_call = litellm_provider_and_mock
        dimension = litellm_provider.get_dimension()
        dummy_embedding_1 = [0.1] * dimension
        dummy_embedding_2 = [0.2] * dimension
        mock_response_data = [
            {"embedding": dummy_embedding_1},
            {"embedding": dummy_embedding_2},
        ]
        mock_response_obj = MagicMock()
        mock_response_obj.data = mock_response_data
        mock_embedding_call.return_value = mock_response_obj
        mocker.patch.object(
            litellm_provider,
            "_validate_embedding_response",
            return_value=mock_response_data,
        )
        extract_mock = mocker.patch.object(
            litellm_provider,
            "_extract_embedding_from_item",
            side_effect=[dummy_embedding_1, dummy_embedding_2],
        )

        texts = ["Text one.", "Text two."]
        embeddings = litellm_provider.embed_texts(texts)

        mock_embedding_call.assert_called_once()
        call_args, call_kwargs = mock_embedding_call.call_args
        assert call_kwargs.get("model") == litellm_provider.config.model_name
        assert call_kwargs.get("input") == texts

        assert embeddings == [dummy_embedding_1, dummy_embedding_2]
        assert extract_mock.call_count == len(texts)

    def test_litellm_provider_error_handling(
        self, litellm_provider_and_mock: MockedProviderFixture, mocker: MockerFixture
    ):
        litellm_provider, mock_embedding_call = litellm_provider_and_mock
        mock_embedding_call.side_effect = litellm.exceptions.APIConnectionError(
            message="Mock connection error",
            llm_provider="litellm",
            model=litellm_provider.config.model_name,
        )

        text = "This will cause an error."
        with pytest.raises(
            EmbeddingProviderError, match="LiteLLM failed to embed text"
        ):
            litellm_provider.embed_text(text)

        mock_embedding_call.side_effect = None
        mock_embedding_call.reset_mock()

        provider_config = litellm_provider.config
        provider_for_init_test = LiteLLMProvider(provider_config)

        mock_embedding_call.side_effect = litellm.exceptions.AuthenticationError(
            message="Mock auth error",
            llm_provider="litellm",
            model=provider_config.model_name,
        )
        _dummy_embedding = [0.0] * provider_config.dimension
        _mock_response_data = [{"embedding": _dummy_embedding}]
        mocker.patch.object(
            provider_for_init_test,
            "_validate_embedding_response",
            return_value=_mock_response_data,
        )
        mocker.patch.object(
            provider_for_init_test,
            "_extract_embedding_from_item",
            return_value=_dummy_embedding,
        )

        call_count_before = mock_embedding_call.call_count

        with pytest.raises(
            EmbeddingProviderError, match="Failed to initialize LiteLLM provider"
        ):
            provider_for_init_test.initialize()

        assert mock_embedding_call.call_count == call_count_before + 1

    def test_litellm_provider_validation_error(
        self, litellm_provider_and_mock: MockedProviderFixture, mocker: MockerFixture
    ):
        litellm_provider, mock_embedding_call = litellm_provider_and_mock
        wrong_dimension = litellm_provider.get_dimension() + 1
        dummy_embedding_wrong = [0.1] * wrong_dimension
        mock_response_data = [{"embedding": dummy_embedding_wrong}]
        mock_response_obj = MagicMock()
        mock_response_obj.data = mock_response_data
        mock_embedding_call.return_value = mock_response_obj
        validate_mock = mocker.patch.object(
            litellm_provider, "_validate_embedding_response"
        )
        validate_mock.side_effect = (
            lambda *args, **kwargs: LiteLLMProvider._validate_embedding_response(
                litellm_provider, *args, **kwargs
            )
        )

        extract_mock = mocker.patch.object(
            litellm_provider, "_extract_embedding_from_item"
        )
        extract_mock.side_effect = (
            lambda *args, **kwargs: LiteLLMProvider._extract_embedding_from_item(
                litellm_provider, *args, **kwargs
            )
        )

        text = "Embed this text."
        with pytest.raises(EmbeddingProviderError, match="Dimension mismatch"):
            litellm_provider.embed_text(text)

        mock_embedding_call.assert_called_once()

    def test_litellm_provider_close(
        self, litellm_provider_and_mock: MockedProviderFixture
    ):
        litellm_provider, _ = litellm_provider_and_mock
        litellm_provider.close()
