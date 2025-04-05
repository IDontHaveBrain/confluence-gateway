import logging
from typing import Optional

from confluence_gateway.core.exceptions import ConfluenceGatewayError
from confluence_gateway.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

logger = logging.getLogger(__name__)


class EmbeddingError(ConfluenceGatewayError):
    """Custom exception for embedding service errors."""

    pass


class EmbeddingService:
    """Service for generating text embeddings using the configured provider."""

    def __init__(self, provider: Optional[EmbeddingProvider]):
        """
        Initialize the EmbeddingService with an embedding provider.

        Args:
            provider: An initialized instance of an EmbeddingProvider, or None if embeddings are disabled.
        """
        self.provider = provider
        if self.provider:
            logger.info(
                f"EmbeddingService initialized with provider: {self.provider.__class__.__name__}"
            )
        else:
            logger.warning(
                "EmbeddingService initialized without a provider. Embedding operations will be disabled."
            )

    def embed_text(self, text: str) -> list[float]:
        """
        Generates embedding for a single piece of text using the configured provider.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the service is not configured with a provider or embedding fails.
        """
        if not self.provider:
            logger.error(
                "Attempted to embed text, but no embedding provider is configured."
            )
            raise EmbeddingError("Embedding provider not configured.")

        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            # Consistent with previous behavior and provider implementations
            return []

        try:
            return self.provider.embed_text(text)
        except EmbeddingProviderError as e:
            logger.error(f"Embedding provider failed to embed text: {e}", exc_info=True)
            raise EmbeddingError("Failed to embed text due to provider error.") from e
        except Exception as e:
            # Catch unexpected errors from the provider call
            logger.error(f"Unexpected error during text embedding: {e}", exc_info=True)
            raise EmbeddingError(
                "An unexpected error occurred during text embedding."
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a batch of texts using the configured provider.

        Args:
            texts: A list of input text strings.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.

        Raises:
            EmbeddingError: If the service is not configured with a provider or embedding fails.
        """
        if not self.provider:
            logger.error(
                "Attempted to embed texts, but no embedding provider is configured."
            )
            raise EmbeddingError("Embedding provider not configured.")

        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

        # Filtering empty/invalid texts is now handled within the provider implementations
        # (SentenceTransformerProvider and LiteLLMProvider already do this)

        try:
            return self.provider.embed_texts(texts)
        except EmbeddingProviderError as e:
            logger.error(
                f"Embedding provider failed to embed batch of texts: {e}", exc_info=True
            )
            raise EmbeddingError(
                "Failed to embed batch of texts due to provider error."
            ) from e
        except Exception as e:
            # Catch unexpected errors from the provider call
            logger.error(
                f"Unexpected error during batch text embedding: {e}", exc_info=True
            )
            raise EmbeddingError(
                "An unexpected error occurred during batch text embedding."
            ) from e

    def get_dimension(self) -> Optional[int]:
        """
        Gets the embedding dimension from the configured provider.

        Returns:
            The embedding dimension as an integer, or None if no provider is configured
            or the provider cannot determine the dimension.
        """
        if not self.provider:
            logger.warning(
                "Attempted to get embedding dimension, but no provider is configured."
            )
            return None

        try:
            return self.provider.get_dimension()
        except EmbeddingProviderError as e:
            # This shouldn't typically happen if the provider initialized correctly,
            # but handle defensively.
            logger.error(
                f"Embedding provider failed to return dimension: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error getting dimension from provider: {e}", exc_info=True
            )
            return None
