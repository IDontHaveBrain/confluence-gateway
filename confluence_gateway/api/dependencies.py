import logging
from typing import Optional

from fastapi import Depends, HTTPException, status

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.core.config import (
    GenerationConfig,
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
    confluence_config,
    embedding_config,
    generation_config,
    indexing_config,
    search_config,
    vector_db_config,
)
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
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


_embedding_provider_instance: Optional[EmbeddingProvider] = None
_embedding_provider_initialized: bool = False


def get_indexing_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    idx_config: IndexingConfig = Depends(lambda: indexing_config),
    srch_config: SearchConfig = Depends(lambda: search_config),
    vdb_config: Optional[VectorDBConfig] = Depends(lambda: vector_db_config),
) -> Optional[IndexingService]:
    if not vdb_config or vdb_config.type == "none":
        if IndexingService._instance:
            logger.warning(
                "Vector DB config changed to 'none' after IndexingService was initialized. Returning existing instance, but it might not function correctly."
            )
            return IndexingService._instance
        logger.warning(
            "Vector DB not configured (type='none'). IndexingService cannot be provided."
        )
        return None

    try:
        instance = IndexingService(
            confluence_client=client,
            indexing_config=idx_config,
            search_config=srch_config,
            vector_db_config=vdb_config,
            embedding_service=embedding_service,
        )
        if not instance.vector_db_adapter or not instance.embedding_service:
            logger.error(
                "IndexingService initialization failed internally (missing adapter or embedding service)."
            )
            return None
        return instance
    except Exception as e:
        logger.error(
            f"Failed to initialize IndexingService singleton: {e}", exc_info=True
        )
        IndexingService._instance = None
        return None


def get_search_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    indexing_service: Optional[IndexingService] = Depends(get_indexing_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db_adapter: Optional[VectorDBAdapter] = Depends(get_vector_db_adapter),
) -> SearchService:
    return SearchService(
        client=client,
        indexing_service=indexing_service,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )


def get_generation_service(
    search_service: SearchService = Depends(get_search_service),
    gen_config: Optional[GenerationConfig] = Depends(lambda: generation_config),
) -> GenerationService:
    if not gen_config or not gen_config.enable:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="RAG generation feature is disabled in the configuration.",
        )

    service = GenerationService(search_service=search_service, config=gen_config)

    if not service.config or not service.config.enable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG generation service could not be initialized (e.g., missing dependencies like litellm). Check server logs.",
        )

    return service
