import logging
from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.embedding.litellm import (
    APIConnectionError,
    AuthenticationError,
    LiteLLMProvider,
    litellm,
)
from confluence_gateway.core.config import EmbeddingConfig
from confluence_gateway.core.exceptions import EmbeddingProviderError

TEST_MODEL_NAME = "test-embedding-model"
TEST_DIMENSION = 128
TEST_API_BASE = "http://localhost:11434"
TEST_API_KEY = "test-key-123"


@pytest.fixture
def valid_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="litellm", model_name=TEST_MODEL_NAME, dimension=TEST_DIMENSION
    )


@pytest.fixture
def config_with_auth() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="litellm",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_DIMENSION,
        litellm_api_base=TEST_API_BASE,
        litellm_api_key=TEST_API_KEY,
    )


@pytest.fixture
def invalid_config_no_model() -> EmbeddingConfig:
    return EmbeddingConfig(provider="litellm", dimension=TEST_DIMENSION)


@pytest.fixture
def invalid_config_no_dimension() -> EmbeddingConfig:
    return EmbeddingConfig(provider="litellm", model_name=TEST_MODEL_NAME)


@pytest.fixture
def mock_litellm_embedding_response():
    mock_response = MagicMock()
    mock_response.data = [
        {"embedding": [0.1] * TEST_DIMENSION, "index": 0, "object": "embedding"}
    ]
    return mock_response


@pytest.fixture
def mock_litellm_batch_embedding_response():
    mock_response = MagicMock()
    mock_response.data = [
        {"embedding": [0.1] * TEST_DIMENSION, "index": 0, "object": "embedding"},
        {"embedding": [0.2] * TEST_DIMENSION, "index": 1, "object": "embedding"},
    ]
    return mock_response


class TestLiteLLMProviderInitialization:
    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_success(
        self, mock_embedding, valid_config, mock_litellm_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        provider = LiteLLMProvider(valid_config)
        provider.initialize()

        assert provider.config == valid_config
        mock_embedding.assert_called_once_with(
            model=TEST_MODEL_NAME,
            input=["validate provider initialization"],
            api_key=None,
            api_base=None,
        )

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_with_auth_success(
        self, mock_embedding, config_with_auth, mock_litellm_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        provider = LiteLLMProvider(config_with_auth)
        provider.initialize()

        assert provider.config == config_with_auth
        mock_embedding.assert_called_once_with(
            model=TEST_MODEL_NAME,
            input=["validate provider initialization"],
            api_key=TEST_API_KEY,
            api_base=TEST_API_BASE,
        )

    def test_initialize_missing_model_name(self, invalid_config_no_model):
        with pytest.raises(
            EmbeddingProviderError, match="requires a model name in the configuration"
        ):
            LiteLLMProvider(invalid_config_no_model)

    def test_initialize_missing_dimension(self, invalid_config_no_dimension):
        with pytest.raises(
            EmbeddingProviderError,
            match="requires an embedding dimension in the configuration",
        ):
            LiteLLMProvider(invalid_config_no_dimension)

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_api_error(self, mock_embedding, valid_config):
        mock_embedding.side_effect = APIConnectionError("Connection failed")
        provider = LiteLLMProvider(valid_config)
        with pytest.raises(EmbeddingProviderError, match="APIConnectionError"):
            provider.initialize()

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_auth_error(self, mock_embedding, valid_config):
        mock_embedding.side_effect = AuthenticationError("Invalid key")
        provider = LiteLLMProvider(valid_config)
        with pytest.raises(EmbeddingProviderError, match="AuthenticationError"):
            provider.initialize()

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_invalid_response_format(self, mock_embedding, valid_config):
        mock_response = MagicMock()
        mock_response.data = []
        mock_embedding.return_value = mock_response
        provider = LiteLLMProvider(valid_config)
        with pytest.raises(
            EmbeddingProviderError, match="Test embedding validation failed"
        ):
            provider.initialize()

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_initialize_dimension_mismatch(self, mock_embedding, valid_config):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * (TEST_DIMENSION - 1)}]
        mock_embedding.return_value = mock_response
        provider = LiteLLMProvider(valid_config)
        with pytest.raises(EmbeddingProviderError, match="Dimension mismatch"):
            provider.initialize()


@patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
class TestLiteLLMProviderEmbedding:
    @pytest.fixture(autouse=True)
    def setup_provider(
        self, mock_embedding, valid_config, mock_litellm_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        self.provider = LiteLLMProvider(valid_config)
        self.provider.initialize()
        mock_embedding.reset_mock()

    def test_embed_text_success(self, mock_embedding, mock_litellm_embedding_response):
        mock_embedding.return_value = mock_litellm_embedding_response
        text = "This is a test sentence."
        embedding = self.provider.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == TEST_DIMENSION
        assert embedding == [0.1] * TEST_DIMENSION
        mock_embedding.assert_called_once_with(
            model=TEST_MODEL_NAME, input=[text], api_key=None, api_base=None
        )

    def test_embed_texts_success(
        self, mock_embedding, mock_litellm_batch_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_batch_embedding_response
        texts = ["First sentence.", "Second sentence."]
        embeddings = self.provider.embed_texts(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * TEST_DIMENSION
        assert embeddings[1] == [0.2] * TEST_DIMENSION
        mock_embedding.assert_called_once_with(
            model=TEST_MODEL_NAME, input=texts, api_key=None, api_base=None
        )

    def test_embed_text_empty(self, mock_embedding, caplog):
        with caplog.at_level(logging.WARNING):
            embedding = self.provider.embed_text("")
        assert embedding == []
        assert "Received empty or invalid text" in caplog.text
        mock_embedding.assert_not_called()

    def test_embed_text_invalid(self, mock_embedding, caplog):
        with caplog.at_level(logging.WARNING):
            embedding = self.provider.embed_text(None)
        assert embedding == []
        assert "Received empty or invalid text" in caplog.text
        mock_embedding.assert_not_called()

    def test_embed_texts_empty_list(self, mock_embedding, caplog):
        with caplog.at_level(logging.WARNING):
            embeddings = self.provider.embed_texts([])
        assert embeddings == []
        assert "Received empty list for batch embedding" in caplog.text
        mock_embedding.assert_not_called()

    def test_embed_texts_with_empty_invalid_strings(
        self, mock_embedding, mock_litellm_embedding_response, caplog
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        texts = ["Valid sentence.", "", "Another valid one.", None, "  "]
        valid_texts = ["Valid sentence.", "Another valid one."]

        with caplog.at_level(logging.WARNING):
            embeddings = self.provider.embed_texts(texts)

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1] * TEST_DIMENSION
        assert "Filtered out 3 empty/invalid strings" in caplog.text
        mock_embedding.assert_called_once_with(
            model=TEST_MODEL_NAME, input=valid_texts, api_key=None, api_base=None
        )

    def test_embed_texts_all_empty_invalid(self, mock_embedding, caplog):
        texts = ["", None, "   "]
        with caplog.at_level(logging.WARNING):
            embeddings = self.provider.embed_texts(texts)
        assert embeddings == []
        assert "All texts in the batch were empty or invalid" in caplog.text
        mock_embedding.assert_not_called()

    def test_embed_text_api_error(self, mock_embedding):
        mock_embedding.side_effect = APIConnectionError("Connection failed")
        with pytest.raises(EmbeddingProviderError, match="APIConnectionError"):
            self.provider.embed_text("test")

    def test_embed_texts_api_error(self, mock_embedding):
        mock_embedding.side_effect = APIConnectionError("Connection failed")
        with pytest.raises(EmbeddingProviderError, match="APIConnectionError"):
            self.provider.embed_texts(["test1", "test2"])

    def test_embed_text_invalid_response(self, mock_embedding):
        mock_response = MagicMock()
        mock_response.data = []
        mock_embedding.return_value = mock_response
        with pytest.raises(
            EmbeddingProviderError, match="LiteLLM embedding response data mismatch"
        ):
            self.provider.embed_text("test")

    def test_embed_texts_invalid_response_count(self, mock_embedding):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * TEST_DIMENSION}]
        mock_embedding.return_value = mock_response
        with pytest.raises(
            EmbeddingProviderError, match="LiteLLM embedding response data mismatch"
        ):
            self.provider.embed_texts(["test1", "test2"])

    def test_embed_texts_invalid_response_dimension(self, mock_embedding):
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * (TEST_DIMENSION - 1)}]
        mock_embedding.return_value = mock_response
        with pytest.raises(EmbeddingProviderError, match="Dimension mismatch"):
            self.provider.embed_texts(["test1"])


class TestLiteLLMProviderMisc:
    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_get_dimension_success(
        self, mock_embedding, valid_config, mock_litellm_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        provider = LiteLLMProvider(valid_config)
        provider.initialize()
        assert provider.get_dimension() == TEST_DIMENSION

    def test_get_dimension_not_configured(self):
        config_no_dim = EmbeddingConfig(provider="litellm", model_name="test")
        with pytest.raises(
            EmbeddingProviderError,
            match="requires an embedding dimension in the configuration",
        ):
            LiteLLMProvider(config_no_dim)

    @patch("confluence_gateway.adapters.embedding.litellm.litellm.embedding")
    def test_close_success(
        self, mock_embedding, valid_config, mock_litellm_embedding_response
    ):
        mock_embedding.return_value = mock_litellm_embedding_response
        provider = LiteLLMProvider(valid_config)
        provider.initialize()
        provider.close()

    def test_close_before_initialize(self, valid_config):
        provider = LiteLLMProvider(valid_config)
        provider.close()
