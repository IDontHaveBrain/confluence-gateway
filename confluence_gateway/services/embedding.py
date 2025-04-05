import logging
from typing import Optional

from confluence_gateway.core.exceptions import ConfluenceGatewayError
from confluence_gateway.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

logger = logging.getLogger(__name__)


class EmbeddingError(ConfluenceGatewayError):
    pass


class EmbeddingService:
    def __init__(self, provider: Optional[EmbeddingProvider]):
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
        if not self.provider:
            logger.error(
                "Attempted to embed text, but no embedding provider is configured."
            )
            raise EmbeddingError("Embedding provider not configured.")

        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []

        try:
            return self.provider.embed_text(text)
        except EmbeddingProviderError as e:
            logger.error(f"Embedding provider failed to embed text: {e}", exc_info=True)
            raise EmbeddingError("Failed to embed text due to provider error.") from e
        except Exception as e:
            logger.error(f"Unexpected error during text embedding: {e}", exc_info=True)
            raise EmbeddingError(
                "An unexpected error occurred during text embedding."
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
            logger.error(
                f"Unexpected error during batch text embedding: {e}", exc_info=True
            )
            raise EmbeddingError(
                "An unexpected error occurred during batch text embedding."
            ) from e

    def get_dimension(self) -> Optional[int]:
        if not self.provider:
            logger.warning(
                "Attempted to get embedding dimension, but no provider is configured."
            )
            return None

        try:
            return self.provider.get_dimension()
        except EmbeddingProviderError as e:
            logger.error(
                f"Embedding provider failed to return dimension: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error getting dimension from provider: {e}", exc_info=True
            )
            return None
