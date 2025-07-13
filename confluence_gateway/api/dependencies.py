import logging
from threading import Lock

from fastapi import Depends, HTTPException, status

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.factory import (
    get_vector_db_adapter as _get_vector_db_adapter_factory,
)
from confluence_gateway.core.config import (
    GenerationConfig,
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
    confluence_config,
    embedding_config,
    generation_config,
    indexing_config,
    is_dev_mode,
    search_config,
    vector_db_config,
)
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService

logger = logging.getLogger(__name__)

# Global singleton instances for API caching
_confluence_client_instance: ConfluenceClient | None = None
_confluence_client_initialized: bool = False
_confluence_client_lock = Lock()

_vector_db_adapter_instance: VectorDBAdapter | None = None
_vector_db_adapter_initialized: bool = False
_vector_db_adapter_lock = Lock()

_embedding_service_instance: EmbeddingService | None = None
_embedding_service_initialized: bool = False
_embedding_service_lock = Lock()

_search_service_instance: SearchService | None = None
_search_service_initialized: bool = False
_search_service_lock = Lock()

_generation_service_instance: GenerationService | None = None
_generation_service_initialized: bool = False
_generation_service_lock = Lock()

_embedding_provider_instance: EmbeddingProvider | None = None
_embedding_provider_initialized: bool = False

_indexing_service_instance: IndexingService | None = None
_indexing_service_lock = Lock()


def get_confluence_client() -> ConfluenceClient:
    global _confluence_client_instance
    global _confluence_client_initialized

    if _confluence_client_initialized:
        if _confluence_client_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Confluence client failed to initialize previously. Check configuration.",
            )
        return _confluence_client_instance

    with _confluence_client_lock:
        if _confluence_client_initialized:
            if _confluence_client_instance is None:  # type: ignore[unreachable]
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Confluence client failed to initialize previously. Check configuration.",
                )
            return _confluence_client_instance

        if not confluence_config:
            logger.error("Confluence configuration is missing or invalid.")
            _confluence_client_initialized = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Confluence configuration is missing or invalid.",
            )

        try:
            logger.info("Attempting to initialize Confluence client singleton...")
            _confluence_client_instance = ConfluenceClient(config=confluence_config)
            _confluence_client_initialized = True
            logger.info("Confluence client singleton initialized successfully.")
            return _confluence_client_instance
        except Exception as e:
            logger.error(
                f"Failed to initialize Confluence client singleton: {e}",
                exc_info=True,
            )
            _confluence_client_instance = None
            _confluence_client_initialized = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize Confluence client: {e}",
            )


def get_embedding_provider_dependency() -> EmbeddingProvider | None:
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


def get_vector_db_adapter() -> VectorDBAdapter | None:
    global _vector_db_adapter_instance
    global _vector_db_adapter_initialized

    if _vector_db_adapter_initialized:
        return _vector_db_adapter_instance

    with _vector_db_adapter_lock:
        if _vector_db_adapter_initialized:
            return _vector_db_adapter_instance  # type: ignore[unreachable]

        try:
            logger.info("Attempting to initialize Vector DB adapter singleton...")
            _vector_db_adapter_instance = _get_vector_db_adapter_factory()
            _vector_db_adapter_initialized = True

            if _vector_db_adapter_instance:
                logger.info(
                    f"Vector DB adapter singleton ({_vector_db_adapter_instance.__class__.__name__}) initialized successfully."
                )
            else:
                logger.info("Vector DB adapter singleton is None (may be disabled).")

            return _vector_db_adapter_instance
        except Exception as e:
            logger.error(
                f"Failed to initialize Vector DB adapter singleton: {e}",
                exc_info=True,
            )
            _vector_db_adapter_instance = None
            _vector_db_adapter_initialized = True
            return None


def get_embedding_service(
    provider: EmbeddingProvider | None = Depends(get_embedding_provider_dependency),
) -> EmbeddingService:
    global _embedding_service_instance
    global _embedding_service_initialized

    if _embedding_service_initialized:
        if _embedding_service_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding service instance is unexpectedly None",
            )
        return _embedding_service_instance

    with _embedding_service_lock:
        if _embedding_service_initialized:
            if _embedding_service_instance is None:  # type: ignore[unreachable]
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Embedding service instance is unexpectedly None",
                )
            return _embedding_service_instance

        try:
            logger.info("Attempting to initialize EmbeddingService singleton...")
            _embedding_service_instance = EmbeddingService(provider=provider)
            _embedding_service_initialized = True
            logger.info("EmbeddingService singleton initialized successfully.")
            return _embedding_service_instance
        except Exception as e:
            logger.error(
                f"Failed to initialize EmbeddingService singleton: {e}",
                exc_info=True,
            )
            _embedding_service_instance = None
            _embedding_service_initialized = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize EmbeddingService: {e}",
            )


def get_indexing_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db_adapter: VectorDBAdapter | None = Depends(get_vector_db_adapter),
    idx_config: IndexingConfig = Depends(lambda: indexing_config),
    srch_config: SearchConfig = Depends(lambda: search_config),
    vdb_config: VectorDBConfig | None = Depends(lambda: vector_db_config),
) -> IndexingService | None:
    global _indexing_service_instance

    if _indexing_service_instance is not None:
        return _indexing_service_instance

    with _indexing_service_lock:
        if _indexing_service_instance is None:
            if not vector_db_adapter:
                logger.warning(
                    "Vector DB Adapter not available. IndexingService cannot be provided."
                )
                return None
            if not vdb_config:
                logger.warning(
                    "Vector DB Config not available (needed for chunk settings). IndexingService cannot be provided."
                )
                return None
            try:
                logger.info("Attempting to initialize IndexingService singleton...")
                instance = IndexingService(
                    confluence_client=client,
                    indexing_config=idx_config,
                    search_config=srch_config,
                    vector_db_adapter=vector_db_adapter,
                    embedding_service=embedding_service,
                )
                if not instance.vector_db_adapter:
                    logger.error(
                        "IndexingService initialization failed unexpectedly (adapter became None post-init)."
                    )
                    return None
                _indexing_service_instance = instance
                logger.info("IndexingService singleton initialized successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize IndexingService singleton: {e}",
                    exc_info=True,
                )
                _indexing_service_instance = None
                return None

    return _indexing_service_instance


def get_search_service(
    client: ConfluenceClient = Depends(get_confluence_client),
    indexing_service: IndexingService | None = Depends(get_indexing_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_db_adapter: VectorDBAdapter | None = Depends(get_vector_db_adapter),
) -> SearchService:
    global _search_service_instance
    global _search_service_initialized

    if _search_service_initialized:
        if _search_service_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service instance is unexpectedly None",
            )
        return _search_service_instance

    with _search_service_lock:
        if _search_service_initialized:
            if _search_service_instance is None:  # type: ignore[unreachable]
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Search service instance is unexpectedly None",
                )
            return _search_service_instance

        try:
            logger.info("Attempting to initialize SearchService singleton...")
            _search_service_instance = SearchService(
                client=client,
                indexing_service=indexing_service,
                embedding_service=embedding_service,
                vector_db_adapter=vector_db_adapter,
            )
            _search_service_initialized = True
            logger.info("SearchService singleton initialized successfully.")
            return _search_service_instance
        except Exception as e:
            logger.error(
                f"Failed to initialize SearchService singleton: {e}",
                exc_info=True,
            )
            _search_service_instance = None
            _search_service_initialized = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize SearchService: {e}",
            )


def get_generation_service(
    search_service: SearchService = Depends(get_search_service),
    gen_config: GenerationConfig | None = Depends(lambda: generation_config),
) -> GenerationService:
    global _generation_service_instance
    global _generation_service_initialized

    # Always check configuration first before attempting to use cached instance
    if not gen_config or not gen_config.enable:
        dev_mode_notice = (
            " (DEV MODE: using stub implementation)" if is_dev_mode() else ""
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"RAG generation feature is disabled in the configuration{dev_mode_notice}.",
        )

    if _generation_service_initialized:
        if _generation_service_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Generation service failed to initialize previously. Check server logs.",
            )
        return _generation_service_instance

    with _generation_service_lock:
        if _generation_service_initialized:
            if _generation_service_instance is None:  # type: ignore[unreachable]
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Generation service failed to initialize previously. Check server logs.",
                )
            return _generation_service_instance

        try:
            logger.info("Attempting to initialize GenerationService singleton...")
            service = GenerationService(
                search_service=search_service, config=gen_config
            )

            # In dev mode, allow service even if not fully configured since it uses stubs
            if not is_dev_mode() and (not service.config or not service.config.enable):
                logger.error("GenerationService config is None or disabled after init.")
                _generation_service_instance = None
                _generation_service_initialized = True
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG generation service could not be initialized (e.g., missing dependencies like litellm). Check server logs.",
                )

            _generation_service_instance = service
            _generation_service_initialized = True
            logger.info("GenerationService singleton initialized successfully.")
            return service
        except Exception as e:
            logger.error(
                f"Failed to initialize GenerationService singleton: {e}",
                exc_info=True,
            )
            _generation_service_instance = None
            _generation_service_initialized = True
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize GenerationService: {e}",
            )
