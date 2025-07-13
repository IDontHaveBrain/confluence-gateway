"""Essential integration tests for embedding provider functionality.

Basic tests for core embedding providers:
- SentenceTransformers basic functionality
- LiteLLM basic functionality
- Factory creation
"""

import logging
from unittest.mock import Mock, patch

import pytest
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Test data constants
TEST_TEXT_SHORT = "test embedding"


@pytest.fixture
def sentence_transformers_config() -> EmbeddingConfig:
    """Create SentenceTransformers embedding configuration for testing."""
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        device="cpu",
    )


@pytest.fixture
def litellm_config() -> EmbeddingConfig:
    """Create LiteLLM embedding configuration for testing with mock API."""
    return EmbeddingConfig(
        provider="litellm",
        model_name="text-embedding-ada-002",
        dimension=1536,
        litellm_api_key="test-key-mock",
    )


@pytest.fixture
def mock_sentence_transformers():
    """Mock SentenceTransformers for testing without requiring model download."""
    with patch(
        "confluence_gateway.adapters.embedding.sentence_transformer._get_sentence_transformer_class"
    ) as mock_class:
        mock_instance = Mock()

        def mock_encode(texts, convert_to_numpy=False, show_progress_bar=True):
            result = [0.1] * 384
            if convert_to_numpy:
                return result
            mock_tensor = Mock()
            mock_tensor.tolist.return_value = result
            mock_tensor.__len__ = Mock(return_value=384)
            return mock_tensor

        mock_instance.encode.side_effect = mock_encode
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        mock_transformer = Mock()
        mock_transformer.return_value = mock_instance
        mock_class.return_value = mock_transformer

        with patch(
            "confluence_gateway.adapters.embedding.sentence_transformer._check_torch_available",
            return_value=True,
        ):
            yield mock_transformer


class TestEmbeddingProviderFactory:
    """Test suite for embedding provider factory functionality."""

    def test_factory_returns_none_for_none_config(self):
        """Test that factory returns None when config parameter is None."""
        provider = get_embedding_provider(None)
        assert provider is None


class TestSentenceTransformersProvider:
    """Test suite for SentenceTransformers embedding provider."""

    def test_sentence_transformers_basic_functionality(
        self, sentence_transformers_config, mock_sentence_transformers
    ):
        """Test basic SentenceTransformers provider functionality."""
        provider = get_embedding_provider(sentence_transformers_config)

        assert provider is not None, (
            "SentenceTransformers provider should be created successfully"
        )
        assert hasattr(provider, "embed_text")
        assert provider.config.provider == "sentence-transformers"
        assert provider.config.model_name == "all-MiniLM-L6-v2"
        assert provider.config.dimension == 384

        # Test single text embedding
        embedding = provider.embed_text(TEST_TEXT_SHORT)
        assert isinstance(embedding, list)
        assert len(embedding) == sentence_transformers_config.dimension
        assert all(isinstance(x, float) for x in embedding)


class TestLiteLLMProvider:
    """Test suite for LiteLLM embedding provider."""

    @patch("confluence_gateway.adapters.embedding.litellm._get_litellm")
    def test_litellm_basic_functionality(self, mock_litellm, litellm_config):
        """Test basic LiteLLM provider functionality."""
        mock_litellm_module = Mock()
        mock_response = Mock()
        mock_response.data = [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 1536}
        ]
        mock_litellm_module.embedding.return_value = mock_response
        mock_litellm.return_value = (mock_litellm_module, {}, Mock())

        provider = get_embedding_provider(litellm_config)

        assert provider is not None, "LiteLLM provider should be created successfully"
        assert hasattr(provider, "embed_text")
        assert provider.config.provider == "litellm"
        assert provider.config.model_name == "text-embedding-ada-002"
        assert provider.config.dimension == 1536

        # Test single text embedding
        embedding = provider.embed_text(TEST_TEXT_SHORT)
        assert isinstance(embedding, list)
        assert len(embedding) == litellm_config.dimension
        assert all(isinstance(x, float) for x in embedding)
