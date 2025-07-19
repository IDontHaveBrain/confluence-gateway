import logging

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.core.exceptions import (
    EmbeddingCompatibilityError,
    EmbeddingError,
    EmbeddingProviderError,
)
from confluence_gateway.core.model_info import (
    ModelCompatibilityValidator,
    ModelInfo,
)
from confluence_gateway.services.common.initialization_logger import (
    InitializationLogger,
)
from confluence_gateway.services.common.validation_utils import ValidationUtils

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None):
        self.provider = provider
        self._cached_model_info: ModelInfo | None = None

        # Log provider availability using standardized logging
        InitializationLogger.log_component_availability(
            "EmbeddingService",
            f"provider ({self.provider.__class__.__name__})"
            if self.provider
            else "provider",
            self.provider is not None,
            logger,
            impact_message="Embedding operations will be disabled.",
        )

    def embed_text(self, text: str) -> list[float]:
        if not ValidationUtils.validate_service_dependency(
            self.provider,
            "embedding provider",
            logger,
            required=True,
            operation_name="text embedding",
        ):
            raise EmbeddingError("Embedding provider not configured.")

        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []

        # Provider guaranteed to be not None by validation above
        assert self.provider is not None

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
        if not ValidationUtils.validate_service_dependency(
            self.provider,
            "embedding provider",
            logger,
            required=True,
            operation_name="batch text embedding",
        ):
            raise EmbeddingError("Embedding provider not configured.")

        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

        # Provider guaranteed to be not None by validation above
        assert self.provider is not None

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

    def get_dimension(self) -> int | None:
        if not ValidationUtils.validate_service_dependency(
            self.provider,
            "embedding provider",
            logger,
            operation_name="dimension retrieval",
        ):
            return None

        # Provider guaranteed to be not None by validation above
        assert self.provider is not None

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

    def get_model_info(self) -> ModelInfo | None:
        """
        Get information about the current embedding model.

        Returns:
            ModelInfo if provider is available, None otherwise
        """
        if not self.provider:
            return None

        if self._cached_model_info is None:
            try:
                provider_name = getattr(self.provider.config, "provider", "unknown")
                self._cached_model_info = ModelInfo.from_provider(
                    self.provider, provider_name
                )
            except Exception as e:
                logger.error(f"Failed to create model info: {e}", exc_info=True)
                return None

        return self._cached_model_info

    def validate_compatibility_with_vector_db(
        self, vector_db_adapter: VectorDBAdapter, operation_type: str = "search"
    ) -> None:
        """
        Validate embedding model compatibility with existing vector DB data.

        Args:
            vector_db_adapter: The vector database adapter to check against
            operation_type: Type of operation ("search" or "index")

        Raises:
            EmbeddingCompatibilityError: If models are incompatible
            EmbeddingError: If validation fails due to other errors
        """
        if not ValidationUtils.validate_service_dependency(
            self.provider,
            "embedding provider",
            logger,
            operation_name="compatibility validation",
        ):
            logger.debug("Skipping compatibility validation")
            return

        try:
            # Check if the collection has existing data
            has_data = vector_db_adapter.has_data()

            if not has_data:
                logger.debug(
                    f"Skipping compatibility validation: operation={operation_type}, has_data={has_data}"
                )
                return

            logger.info(
                f"Validating embedding model compatibility for {operation_type} operation..."
            )

            # Get current model info
            current_model_info = self.get_model_info()
            if not current_model_info:
                raise EmbeddingError(
                    "Could not determine current embedding model information"
                )

            # Try to get stored model info from collection metadata
            collection_metadata = vector_db_adapter.get_collection_metadata()
            stored_model_info = None

            if collection_metadata:
                stored_model_info = ModelInfo.from_metadata_dict(collection_metadata)

            # If no stored model info in collection metadata, try to get from document metadata
            if not stored_model_info:
                logger.debug(
                    "No embedding model info in collection metadata, checking document metadata..."
                )
                document_metadata = vector_db_adapter.search_by_metadata(
                    filters={},
                    select=[
                        "embedding_model_name",
                        "embedding_provider",
                        "embedding_dimension",
                        "embedding_model_created_at",
                        "embedding_model_version",
                    ],
                    limit=1,
                )
                validator = ModelCompatibilityValidator()
                stored_model_info = validator.extract_model_info_from_metadata(
                    document_metadata
                )

            if stored_model_info:
                # Use the new ModelCompatibilityValidator
                validator = ModelCompatibilityValidator()
                validator.validate_compatibility(
                    current_model_info, stored_model_info, operation_type
                )

                logger.info("Embedding model compatibility validation passed")
            else:
                logger.info(
                    "No existing embedding model info found, validation skipped"
                )

        except EmbeddingCompatibilityError:
            # Re-raise compatibility errors as-is
            raise
        except Exception as e:
            logger.error(
                f"Embedding compatibility validation failed: {e}", exc_info=True
            )
            raise EmbeddingError(
                f"Failed to validate embedding compatibility: {e}"
            ) from e

    def store_model_info_in_vector_db(self, vector_db_adapter: VectorDBAdapter) -> None:
        """
        Store current embedding model information in the vector database.

        Args:
            vector_db_adapter: The vector database adapter to store info in
        """
        if not ValidationUtils.validate_service_dependency(
            self.provider,
            "embedding provider",
            logger,
            operation_name="model info storage",
        ):
            logger.debug("Skipping model info storage")
            return

        try:
            model_info = self.get_model_info()
            if not model_info:
                logger.warning("Could not get current model info to store")
                return

            # Store in collection metadata
            logger.info(
                f"Storing embedding model info in collection metadata: {model_info.model_name}"
            )
            vector_db_adapter.set_collection_metadata(model_info.to_metadata_dict())

        except Exception as e:
            logger.error(f"Failed to store embedding model info: {e}", exc_info=True)
            # Don't raise here as this is not critical for the main operation
