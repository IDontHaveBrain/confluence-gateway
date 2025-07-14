"""Essential integration tests for embedding provider functionality.

Basic tests for core embedding providers with shared model optimization:
- SentenceTransformers basic functionality (with shared model optimization)
- LiteLLM basic functionality
- Factory creation
- Performance validation for optimization impact
"""

import logging
import time
from unittest.mock import Mock, patch

import pytest
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.core.config import EmbeddingConfig

from tests.fixtures.shared_embedding import (
    inject_shared_model_into_provider,
    log_embedding_operation,
)

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
    """Test suite for SentenceTransformers embedding provider with shared model optimization."""

    def test_sentence_transformers_basic_functionality(
        self, sentence_transformers_config, mock_sentence_transformers
    ):
        """Test basic SentenceTransformers provider functionality."""
        start_time = time.time()

        provider = get_embedding_provider(sentence_transformers_config)

        assert provider is not None, (
            "SentenceTransformers provider should be created successfully"
        )
        assert hasattr(provider, "embed_text")
        assert provider.config.provider == "sentence-transformers"
        assert provider.config.model_name == "all-MiniLM-L6-v2"
        assert provider.config.dimension == 384

        # Test single text embedding
        embedding_start = time.time()
        embedding = provider.embed_text(TEST_TEXT_SHORT)
        embedding_time = time.time() - embedding_start

        assert isinstance(embedding, list)
        assert len(embedding) == sentence_transformers_config.dimension
        assert all(isinstance(x, float) for x in embedding)

        # Log performance metrics
        total_time = time.time() - start_time
        log_embedding_operation("sentence_transformers_basic_test", total_time)
        log_embedding_operation("sentence_transformers_embedding", embedding_time)

        print(
            f"SentenceTransformers provider test completed in {total_time:.3f}s (embedding: {embedding_time:.3f}s)"
        )

    def test_sentence_transformers_with_shared_model(
        self, sentence_transformers_config, shared_sentence_transformer_model
    ):
        """Test SentenceTransformers provider with shared model optimization."""
        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for optimization testing")

        start_time = time.time()

        # Create provider without shared model first (to compare)
        standard_provider = get_embedding_provider(sentence_transformers_config)
        standard_creation_time = time.time() - start_time

        # Create provider with shared model optimization
        optimized_start = time.time()
        optimized_provider = get_embedding_provider(sentence_transformers_config)

        # Inject shared model for optimization
        injection_success = inject_shared_model_into_provider(
            optimized_provider, shared_sentence_transformer_model
        )
        optimized_creation_time = time.time() - optimized_start

        assert injection_success, "Shared model injection should succeed"
        assert optimized_provider is not None

        # Test that both providers work
        standard_embedding = standard_provider.embed_text(TEST_TEXT_SHORT)
        optimized_embedding = optimized_provider.embed_text(TEST_TEXT_SHORT)

        assert len(standard_embedding) == len(optimized_embedding)
        assert isinstance(standard_embedding, list)
        assert isinstance(optimized_embedding, list)

        # Log performance comparison
        log_embedding_operation("standard_provider_creation", standard_creation_time)
        log_embedding_operation("optimized_provider_creation", optimized_creation_time)

        print(
            f"Provider creation: Standard {standard_creation_time:.3f}s vs Optimized {optimized_creation_time:.3f}s"
        )

        # The optimized provider should be faster or similar due to shared model
        if optimized_creation_time < standard_creation_time:
            improvement = (
                (standard_creation_time - optimized_creation_time)
                / standard_creation_time
            ) * 100
            print(f"Optimization improvement: {improvement:.1f}% faster creation time")


class TestLiteLLMProvider:
    """Test suite for LiteLLM embedding provider (non-optimized)."""

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
