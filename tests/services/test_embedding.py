import pytest
from unittest.mock import Mock, patch

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.exceptions import (
    EmbeddingError,
    EmbeddingProviderError,
)
from confluence_gateway.services.embedding import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture
    def mock_provider(self):
        """Create a mock embedding provider."""
        provider = Mock(spec=EmbeddingProvider)
        provider.embed_text.return_value = [0.1, 0.2, 0.3]
        provider.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        provider.get_dimension.return_value = 384
        return provider

    @pytest.fixture
    def service(self, mock_provider):
        """Create an EmbeddingService with mock provider."""
        return EmbeddingService(mock_provider)

    @pytest.fixture
    def service_no_provider(self):
        """Create an EmbeddingService without a provider."""
        return EmbeddingService(None)

    def test_embed_text_success(self, service, mock_provider):
        """Test successful text embedding."""
        text = "This is a test document."
        result = service.embed_text(text)

        assert result == [0.1, 0.2, 0.3]
        mock_provider.embed_text.assert_called_once_with(text)

    def test_embed_text_empty_input(self, service):
        """Test embedding with empty text input."""
        result = service.embed_text("")
        assert result == []

        result = service.embed_text(None)
        assert result == []

    def test_embed_text_unicode_handling(self, service, mock_provider):
        """Test embedding with unicode characters."""
        text = "Unicode test: 你好世界 🌍 Привет мир"
        mock_provider.embed_text.return_value = [0.7, 0.8, 0.9]

        result = service.embed_text(text)

        assert result == [0.7, 0.8, 0.9]
        mock_provider.embed_text.assert_called_once_with(text)

    def test_embed_texts_batch_processing(self, service, mock_provider):
        """Test batch text embedding."""
        texts = ["First document", "Second document"]
        result = service.embed_texts(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_provider.embed_texts.assert_called_once_with(texts)

    def test_embed_texts_large_batch(self, service, mock_provider):
        """Test embedding a large batch of texts."""
        texts = [f"Document {i}" for i in range(100)]
        expected_embeddings = [
            [float(i), float(i + 1), float(i + 2)] for i in range(100)
        ]
        mock_provider.embed_texts.return_value = expected_embeddings

        result = service.embed_texts(texts)

        assert result == expected_embeddings
        assert len(result) == 100
        mock_provider.embed_texts.assert_called_once_with(texts)

    def test_embed_texts_empty_list(self, service):
        """Test batch embedding with empty list."""
        result = service.embed_texts([])
        assert result == []

    def test_get_dimension(self, service, mock_provider):
        """Test getting embedding dimension."""
        result = service.get_dimension()

        assert result == 384
        mock_provider.get_dimension.assert_called_once()

    def test_no_provider_error(self, service_no_provider):
        """Test error when no provider is configured."""
        with pytest.raises(EmbeddingError, match="Embedding provider not configured"):
            service_no_provider.embed_text("test")

        with pytest.raises(EmbeddingError, match="Embedding provider not configured"):
            service_no_provider.embed_texts(["test"])

        # get_dimension returns None instead of raising
        assert service_no_provider.get_dimension() is None

    def test_provider_exception_propagation(self, service, mock_provider):
        """Test that provider exceptions are properly wrapped."""
        # Test EmbeddingProviderError in embed_text
        mock_provider.embed_text.side_effect = EmbeddingProviderError("Provider failed")

        with pytest.raises(
            EmbeddingError, match="Failed to embed text due to provider error"
        ):
            service.embed_text("test")

        # Test EmbeddingProviderError in embed_texts
        mock_provider.embed_texts.side_effect = EmbeddingProviderError(
            "Provider batch failed"
        )

        with pytest.raises(
            EmbeddingError, match="Failed to embed batch of texts due to provider error"
        ):
            service.embed_texts(["test1", "test2"])

        # Test generic exception in embed_text
        mock_provider.embed_text.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(
            EmbeddingError, match="An unexpected error occurred during text embedding"
        ):
            service.embed_text("test")

    def test_rate_limiting_handling(self, service, mock_provider):
        """Test handling of rate limiting errors from provider."""
        # Simulate rate limiting error
        mock_provider.embed_text.side_effect = EmbeddingProviderError(
            "Rate limit exceeded"
        )

        with pytest.raises(EmbeddingError) as exc_info:
            service.embed_text("test")

        assert "Failed to embed text due to provider error" in str(exc_info.value)

    def test_get_dimension_error_handling(self, service, mock_provider):
        """Test error handling in get_dimension."""
        # Test EmbeddingProviderError
        mock_provider.get_dimension.side_effect = EmbeddingProviderError(
            "Cannot get dimension"
        )
        assert service.get_dimension() is None

        # Test generic exception
        mock_provider.get_dimension.side_effect = RuntimeError("Unexpected error")
        assert service.get_dimension() is None

    def test_invalid_input_types(self, service):
        """Test handling of invalid input types."""
        # Non-string input to embed_text
        assert service.embed_text(123) == []
        assert service.embed_text(["list"]) == []
        assert service.embed_text({"dict": "value"}) == []

    def test_mixed_empty_texts_in_batch(self, service, mock_provider):
        """Test batch embedding with mix of empty and non-empty texts."""
        texts = ["Valid text", "", "Another valid text", None, "Final text"]
        # Provider would handle filtering, we test service passes it through
        mock_provider.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        result = service.embed_texts(texts)
        assert len(result) == 3
        mock_provider.embed_texts.assert_called_once_with(texts)

    @patch("confluence_gateway.services.embedding.logger")
    def test_logging_initialization(self, mock_logger):
        """Test proper logging during initialization."""
        # With provider
        provider = Mock(spec=EmbeddingProvider)
        service = EmbeddingService(provider)
        mock_logger.info.assert_called_with(
            f"EmbeddingService initialized with provider: {provider.__class__.__name__}"
        )

        # Without provider
        service = EmbeddingService(None)
        mock_logger.warning.assert_called_with(
            "EmbeddingService initialized without a provider. Embedding operations will be disabled."
        )

    @patch("confluence_gateway.services.embedding.logger")
    def test_logging_errors(self, mock_logger, service, mock_provider):
        """Test proper error logging."""
        mock_provider.embed_text.side_effect = EmbeddingProviderError("Test error")

        with pytest.raises(EmbeddingError):
            service.embed_text("test")

        mock_logger.error.assert_called()
        assert (
            "Embedding provider failed to embed text"
            in mock_logger.error.call_args[0][0]
        )
