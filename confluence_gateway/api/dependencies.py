import logging
from typing import Optional

from fastapi import Depends, HTTPException, status

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.core.config import (
    confluence_config,
    embedding_config,
    vector_db_config,
)
from confluence_gateway.providers.embedding.base import EmbeddingProvider
from confluence_gateway.providers.embedding.factory import get_embedding_provider
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService

logger = logging.getLogger(__name__)


def get_confluence_client():
    if not confluence_config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Confluence configuration is missing or invalid.",
        )
    return ConfluenceClient(config=confluence_config)


def get_embedding_provider_dependency() -> Optional[EmbeddingProvider]:
    """
    Provides a singleton instance of the EmbeddingProvider.

    Initializes the provider using the factory on the first call.
    Returns None if embeddings are disabled or initialization fails.
    """
    global _embedding_provider_instance
    global _embedding_provider_initialized

    # If initialization was already attempted (successfully or not), return the stored instance
    if _embedding_provider_initialized:
        return _embedding_provider_instance

    # Check if embedding is configured and enabled
    if embedding_config and embedding_config.provider != "none":
        try:
            logger.info("Attempting to initialize EmbeddingProvider singleton...")
            # The factory function handles instantiation AND initialization
            _embedding_provider_instance = get_embedding_provider(embedding_config)

            if _embedding_provider_instance:
                logger.info(
                    f"EmbeddingProvider singleton ({_embedding_provider_instance.__class__.__name__}) initialized successfully."
                )
            else:
                # Factory returned None, likely due to an error during its initialization step
                logger.error(
                    "EmbeddingProvider singleton initialization failed (factory returned None). Check previous logs for details."
                )

        except Exception as e:
            # Catch any unexpected errors during factory call itself
            logger.error(
                f"Critical error during EmbeddingProvider factory call: {e}",
                exc_info=True,
            )
            _embedding_provider_instance = (
                None  # Ensure instance is None on critical failure
            )
    else:
        logger.info(
            "Embedding provider explicitly disabled or configuration missing. Singleton is None."
        )
        _embedding_provider_instance = None  # Explicitly set to None if disabled

    _embedding_provider_initialized = (
        True  # Mark that initialization has been attempted
    )
    return _embedding_provider_instance


def get_embedding_service(
    provider: Optional[EmbeddingProvider] = Depends(get_embedding_provider_dependency),
) -> EmbeddingService:
    """Provides an instance of the EmbeddingService, injecting the provider."""
    # EmbeddingService is lightweight, creating it per-request is acceptable.
    return EmbeddingService(provider=provider)


# Singleton patterns for services to avoid recreating them for each request
_indexing_service_instance: Optional[IndexingService] = None
_embedding_provider_instance: Optional[EmbeddingProvider] = None
_embedding_provider_initialized: bool = False  # Flag to track initialization attempt


def get_indexing_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> Optional[IndexingService]:
    """
    Provides an instance of the IndexingService.
    Returns None if vector DB is not configured/enabled.
    """
    global _indexing_service_instance
    # Initialize only if vector DB is configured and instance doesn't exist
    if (
        _indexing_service_instance is None
        and vector_db_config
        and vector_db_config.type != "none"
    ):
        try:
            _indexing_service_instance = IndexingService(
                confluence_client=client, embedding_service=embedding_service
            )
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"Failed to initialize IndexingService: {e}", exc_info=True
            )
            return None

    return _indexing_service_instance


def get_search_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    indexing_service: Optional[IndexingService] = Depends(get_indexing_service),
) -> SearchService:
    """Provides an instance of the SearchService, injecting dependencies."""
    return SearchService(client=client, indexing_service=indexing_service)
