from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from confluence_gateway.adapters.confluence.client import ConfluenceClient
    from confluence_gateway.adapters.embedding.base import EmbeddingProvider
    from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
    from confluence_gateway.services.embedding import EmbeddingService
    from confluence_gateway.services.generation import GenerationService
    from confluence_gateway.services.indexing import IndexingService
    from confluence_gateway.services.search import SearchService

from confluence_gateway.cli.common import print_status
from confluence_gateway.core.config import (
    confluence_config,
    embedding_config,
    generation_config,
    indexing_config,
    is_dev_mode,
    search_config,
    vector_db_config,
)

logger = logging.getLogger(__name__)


def _check_dev_mode_feature(feature_name: str) -> None:
    """Check if a feature is available when dev mode is enabled."""
    if is_dev_mode():
        print_status(
            f"Notice: {feature_name} is running in DEV MODE - using stub implementation for faster development iteration. "
            f"To access full functionality, run without CONFLUENCE_GATEWAY_DEV_MODE environment variable.",
            "warning",
        )


# Global singleton instances for CLI caching (using forward references)
_confluence_client_instance: ConfluenceClient | None = None
_confluence_client_initialized: bool = False

_embedding_provider_instance: EmbeddingProvider | None = None
_embedding_provider_initialized: bool = False

_vector_db_adapter_instance: VectorDBAdapter | None = None
_vector_db_adapter_initialized: bool = False

_embedding_service_instance: EmbeddingService | None = None
_embedding_service_initialized: bool = False

_search_service_instance: SearchService | None = None
_search_service_initialized: bool = False

_indexing_service_instance: IndexingService | None = None
_indexing_service_initialized: bool = False


def _get_confluence_client() -> ConfluenceClient:
    # Import here to avoid startup overhead
    from confluence_gateway.adapters.confluence.client import ConfluenceClient

    global _confluence_client_instance
    global _confluence_client_initialized

    if _confluence_client_initialized:
        if _confluence_client_instance is None:
            print_status(
                "Error: Confluence client failed to initialize previously. Check configuration.",
                "error",
            )
            raise typer.Exit(code=1)
        return _confluence_client_instance

    if not confluence_config:
        print_status(
            "Error: Confluence configuration (CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN) is missing or invalid.",
            "error",
        )
        _confluence_client_initialized = True
        raise typer.Exit(code=1)

    try:
        logger.info("Initializing Confluence client singleton...")
        _confluence_client_instance = ConfluenceClient(config=confluence_config)
        _confluence_client_initialized = True
        logger.info("Confluence client singleton initialized successfully.")
        return _confluence_client_instance
    except Exception as e:
        print_status(f"Error: Failed to initialize Confluence client: {e}", "error")
        logger.error("Failed to initialize Confluence client", exc_info=True)
        _confluence_client_instance = None
        _confluence_client_initialized = True
        raise typer.Exit(code=1)


def _get_embedding_provider() -> EmbeddingProvider | None:
    # Import here to avoid startup overhead
    from confluence_gateway.adapters.embedding.factory import get_embedding_provider

    global _embedding_provider_instance
    global _embedding_provider_initialized

    if _embedding_provider_initialized:
        return _embedding_provider_instance

    if embedding_config and embedding_config.provider != "none":
        try:
            logger.info("Initializing embedding provider singleton...")
            _embedding_provider_instance = get_embedding_provider(embedding_config)

            if _embedding_provider_instance:
                logger.info(
                    f"Embedding provider singleton ({_embedding_provider_instance.__class__.__name__}) initialized successfully."
                )
            else:
                logger.error(
                    "Embedding provider singleton initialization failed (factory returned None). Check previous logs for details."
                )

        except Exception as e:
            print_status(
                f"Error: Unexpected error getting embedding provider: {e}", "error"
            )
            logger.error(
                f"Critical error during embedding provider factory call: {e}",
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


def _get_embedding_service() -> EmbeddingService:
    # Import here to avoid startup overhead
    from confluence_gateway.services.embedding import EmbeddingService

    global _embedding_service_instance
    global _embedding_service_initialized

    if _embedding_service_initialized:
        if _embedding_service_instance is None:
            # This should not happen since we always create the service
            raise RuntimeError("Embedding service instance is unexpectedly None")
        return _embedding_service_instance

    provider = _get_embedding_provider()
    if provider and is_dev_mode():
        _check_dev_mode_feature("Embedding Service")

    logger.info("Initializing embedding service singleton...")
    _embedding_service_instance = EmbeddingService(provider=provider)
    _embedding_service_initialized = True
    logger.info("Embedding service singleton initialized successfully.")
    return _embedding_service_instance


def _get_vector_db_adapter() -> VectorDBAdapter | None:
    # Import here to avoid startup overhead
    from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter

    global _vector_db_adapter_instance
    global _vector_db_adapter_initialized

    if _vector_db_adapter_initialized:
        return _vector_db_adapter_instance

    try:
        logger.info("Initializing vector DB adapter singleton...")
        _vector_db_adapter_instance = get_vector_db_adapter()
        _vector_db_adapter_initialized = True

        if _vector_db_adapter_instance:
            if is_dev_mode():
                _check_dev_mode_feature("Vector Database")
            logger.info(
                f"Vector DB adapter singleton ({_vector_db_adapter_instance.__class__.__name__}) initialized successfully."
            )
        else:
            logger.info("Vector DB adapter singleton is None (may be disabled).")

        return _vector_db_adapter_instance
    except Exception as e:
        print_status(f"Error: Unexpected error getting vector DB adapter: {e}", "error")
        logger.error("Unexpected error getting vector DB adapter", exc_info=True)
        _vector_db_adapter_instance = None
        _vector_db_adapter_initialized = True
        return None


def _get_search_service() -> SearchService:
    # Import here to avoid startup overhead
    from confluence_gateway.services.search import SearchService

    global _search_service_instance
    global _search_service_initialized

    if _search_service_initialized:
        if _search_service_instance is None:
            # This should not happen since we always create the service
            raise RuntimeError("Search service instance is unexpectedly None")
        return _search_service_instance

    client = _get_confluence_client()
    embedding_service = _get_embedding_service()
    vector_db_adapter = _get_vector_db_adapter()

    logger.info("Initializing search service singleton...")
    _search_service_instance = SearchService(
        client=client,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )
    _search_service_initialized = True
    logger.info("Search service singleton initialized successfully.")
    return _search_service_instance


def _get_indexing_service() -> IndexingService | None:
    # Import here to avoid startup overhead
    from confluence_gateway.services.indexing import IndexingService

    global _indexing_service_instance
    global _indexing_service_initialized

    if _indexing_service_initialized:
        return _indexing_service_instance

    if not vector_db_config or vector_db_config.type == "none":
        print_status(
            "Warning: Vector DB is not configured (VECTOR_DB_TYPE=none). Indexing service is unavailable.",
            "warning",
        )
        _indexing_service_instance = None
        _indexing_service_initialized = True
        return None

    client = _get_confluence_client()
    embedding_service = _get_embedding_service()

    if not embedding_service.provider:
        print_status(
            "Warning: Embedding provider is not configured or failed to initialize. Indexing service is unavailable as it requires embeddings.",
            "warning",
        )
        _indexing_service_instance = None
        _indexing_service_initialized = True
        return None

    vector_db_adapter = _get_vector_db_adapter()
    if not vector_db_adapter:
        print_status(
            "Error: Failed to initialize Vector DB Adapter. Indexing service is unavailable.",
            "error",
        )
        _indexing_service_instance = None
        _indexing_service_initialized = True
        return None

    try:
        logger.info("Initializing indexing service singleton...")
        service = IndexingService(
            confluence_client=client,
            indexing_config=indexing_config,
            search_config=search_config,
            embedding_service=embedding_service,
            vector_db_adapter=vector_db_adapter,
        )
        if not service.vector_db_adapter:
            print_status(
                "Error: IndexingService initialized, but Vector DB Adapter failed to set up internally.",
                "error",
            )
            logger.error("IndexingService vector_db_adapter is None after init.")
            _indexing_service_instance = None
            _indexing_service_initialized = True
            return None

        _indexing_service_instance = service
        _indexing_service_initialized = True
        logger.info("Indexing service singleton initialized successfully.")
        return service
    except Exception as e:
        print_status(f"Error: Failed to initialize Indexing service: {e}", "error")
        logger.error("Failed to initialize Indexing service", exc_info=True)
        _indexing_service_instance = None
        _indexing_service_initialized = True
        return None


def _get_generation_service() -> GenerationService:
    # Import here to avoid startup overhead
    from confluence_gateway.services.generation import GenerationService

    _check_dev_mode_feature("RAG Generation")

    if not generation_config or not generation_config.enable:
        print_status(
            "Error: RAG Generation is disabled in configuration (GENERATION_ENABLE=False).",
            "error",
        )
        raise typer.Exit(code=1)

    search_service = _get_search_service()

    try:
        service = GenerationService(
            search_service=search_service, config=generation_config
        )
        if not is_dev_mode() and (not service.config or not service.config.enable):
            print_status(
                "Error: Generation service could not be properly initialized (e.g., missing dependencies). Check logs.",
                "error",
            )
            logger.error("GenerationService config is None or disabled after init.")
            raise typer.Exit(code=1)
        return service
    except Exception as e:
        print_status(f"Error: Failed to initialize Generation service: {e}", "error")
        logger.error("Failed to initialize Generation service", exc_info=True)
        raise typer.Exit(code=1)
