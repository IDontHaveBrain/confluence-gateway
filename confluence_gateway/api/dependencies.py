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
    global _embedding_provider_instance
    global _embedding_provider_initialized

    if _embedding_provider_initialized:
        return _embedding_provider_instance

    if embedding_config and embedding_config.provider != "none":
        try:
            logger.info("Attempting to initialize EmbeddingProvider singleton...")
            _embedding_provider_instance = get_embedding_provider(embedding_config)

            if _embedding_provider_instance:
                logger.info(
                    f"EmbeddingProvider singleton ({_embedding_provider_instance.__class__.__name__}) initialized successfully."
                )
            else:
                logger.error(
                    "EmbeddingProvider singleton initialization failed (factory returned None). Check previous logs for details."
                )

        except Exception as e:
            logger.error(
                f"Critical error during EmbeddingProvider factory call: {e}",
                exc_info=True,
            )
            _embedding_provider_instance = None
    else:
        logger.info(
            "Embedding provider explicitly disabled or configuration missing. Singleton is None."
        )
        _embedding_provider_instance = None

    _embedding_provider_initialized = True
    return _embedding_provider_instance


def get_embedding_service(
    provider: Optional[EmbeddingProvider] = Depends(get_embedding_provider_dependency),
) -> EmbeddingService:
    return EmbeddingService(provider=provider)


_indexing_service_instance: Optional[IndexingService] = None
_embedding_provider_instance: Optional[EmbeddingProvider] = None
_embedding_provider_initialized: bool = False


def get_indexing_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> Optional[IndexingService]:
    global _indexing_service_instance
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
            logging.getLogger(__name__).error(
                f"Failed to initialize IndexingService: {e}", exc_info=True
            )
            return None

    return _indexing_service_instance


def get_search_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    indexing_service: Optional[IndexingService] = Depends(get_indexing_service),
) -> SearchService:
    return SearchService(client=client, indexing_service=indexing_service)
