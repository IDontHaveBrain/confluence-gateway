"""FastAPI dependencies using ServiceContainer infrastructure."""

from functools import lru_cache
from typing import Any, cast

from fastapi import Depends

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.core.config import get_development_context
from confluence_gateway.core.service_container import APIErrorStrategy, ServiceContainer
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing_service import IndexingService
from confluence_gateway.services.search import SearchService


@lru_cache(maxsize=1)
def get_service_container() -> ServiceContainer:
    """Get singleton ServiceContainer with thread-safe mode for API usage."""
    container = ServiceContainer(thread_safe=True)
    _setup_service_factories(container)
    return container


def _setup_service_factories(container: ServiceContainer) -> None:
    """Setup service factories using direct factory implementations."""
    from confluence_gateway.adapters.embedding.factory import get_embedding_provider
    from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
    from confluence_gateway.core.config import (
        confluence_config,
        embedding_config,
        generation_config,
        indexing_config,
        search_config,
    )
    from confluence_gateway.services.embedding import EmbeddingService
    from confluence_gateway.services.search_modules.search_validator import (
        SearchValidator,
    )

    # Confluence client factory
    def confluence_client_factory() -> ConfluenceClient | None:
        dev_context = get_development_context()
        if dev_context.enabled:
            return None
        return ConfluenceClient(confluence_config)

    # Search service factory
    def search_service_factory() -> SearchService | None:
        dev_context = get_development_context()
        if dev_context.enabled:
            # Create stub search service for dev mode
            return SearchService(
                client=None,
                indexing_service=None,
                embedding_service=None,
                vector_db_adapter=None,
                search_validator=SearchValidator(),
            )

        confluence_client = container.get_service("confluence_client")
        if not confluence_client:
            return None

        # Get dependencies - will return None if unavailable
        vector_db_adapter = (
            get_vector_db_adapter()
            if embedding_config and embedding_config.provider != "none"
            else None
        )
        embedding_provider = (
            get_embedding_provider(embedding_config)
            if embedding_config and embedding_config.provider != "none"
            else None
        )
        embedding_service = (
            EmbeddingService(embedding_provider) if embedding_provider else None
        )
        indexing_service = (
            container.get_service("indexing_service") if vector_db_adapter else None
        )

        search_validator = SearchValidator()
        return SearchService(
            client=confluence_client,
            indexing_service=indexing_service,
            embedding_service=embedding_service,
            vector_db_adapter=vector_db_adapter,
            search_validator=search_validator,
        )

    # Indexing service factory
    def indexing_service_factory() -> IndexingService | Any | None:
        dev_context = get_development_context()

        if dev_context.enabled:
            # Create a stub IndexingService for dev mode
            from confluence_gateway.services.indexing_service import IndexingService

            # Create a minimal stub implementation that can handle status requests
            class StubIndexingService:
                def __init__(self) -> None:
                    self._status = {
                        "status": "idle",
                        "last_run_start_time": None,
                        "last_run_end_time": None,
                        "last_error_message": None,
                    }

                @property
                def status(self) -> dict[str, Any]:
                    return self._status

                async def run_indexing(
                    self, space_keys: list[str] | None = None, index_all: bool = False
                ) -> None:
                    # Mock indexing operation for dev mode
                    import time
                    from datetime import datetime, timezone

                    self._status["status"] = "running"
                    self._status["last_run_start_time"] = datetime.now(
                        timezone.utc
                    ).isoformat()

                    # Simulate quick indexing
                    time.sleep(0.1)

                    self._status["status"] = "success"
                    self._status["last_run_end_time"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    self._status["last_error_message"] = None

            return StubIndexingService()

        confluence_client = container.get_service("confluence_client")
        if not confluence_client:
            return None

        vector_db_adapter = get_vector_db_adapter()
        if not vector_db_adapter:
            return None

        embedding_provider = (
            get_embedding_provider(embedding_config)
            if embedding_config and embedding_config.provider != "none"
            else None
        )
        embedding_service = (
            EmbeddingService(embedding_provider) if embedding_provider else None
        )

        return IndexingService(
            confluence_client=confluence_client,
            indexing_config=indexing_config,
            search_config=search_config,
            embedding_service=embedding_service,
            vector_db_adapter=vector_db_adapter,
        )

    # Generation service factory
    def generation_service_factory() -> GenerationService | None:
        dev_context = get_development_context()

        if dev_context.enabled:
            # In dev mode, create a simple stub search service to avoid circular dependencies
            from confluence_gateway.services.search import SearchService
            from confluence_gateway.services.search_modules.search_validator import (
                SearchValidator,
            )

            # Create a stub search service directly instead of getting from container
            stub_search_service = SearchService(
                client=None,
                indexing_service=None,
                embedding_service=None,
                vector_db_adapter=None,
                search_validator=SearchValidator(),
            )

            return GenerationService(
                search_service=stub_search_service,
                config=None,  # Force None config in dev mode to avoid validation issues
            )

        search_service = container.get_service("search_service")
        if not search_service:
            return None

        return GenerationService(
            search_service=search_service,
            config=generation_config,
        )

    # Register factories
    container.register_factory("confluence_client", confluence_client_factory)
    container.register_factory("search_service", search_service_factory)
    container.register_factory("indexing_service", indexing_service_factory)
    container.register_factory("generation_service", generation_service_factory)


# FastAPI dependency functions
def get_confluence_client(
    container: ServiceContainer = Depends(get_service_container),
) -> ConfluenceClient | None:
    """Get Confluence client dependency."""
    dev_context = get_development_context()
    if dev_context.enabled:
        # In dev mode, allow None client
        return cast(
            ConfluenceClient | None, container.get_service("confluence_client", None)
        )
    else:
        # In production mode, use error strategy
        return cast(
            ConfluenceClient | None,
            container.get_service("confluence_client", None, APIErrorStrategy()),
        )


def get_search_service(
    container: ServiceContainer = Depends(get_service_container),
) -> SearchService:
    """Get search service dependency."""
    return cast(
        SearchService, container.get_service("search_service", None, APIErrorStrategy())
    )


def get_indexing_service(
    container: ServiceContainer = Depends(get_service_container),
) -> IndexingService | None:
    """Get indexing service dependency."""
    return cast(
        IndexingService | None,
        container.get_service("indexing_service", None, APIErrorStrategy()),
    )


def get_generation_service(
    container: ServiceContainer = Depends(get_service_container),
) -> GenerationService:
    """Get generation service dependency."""
    return cast(
        GenerationService,
        container.get_service("generation_service", None, APIErrorStrategy()),
    )


def get_health_status(
    container: ServiceContainer = Depends(get_service_container),
) -> dict[str, Any]:
    """Get comprehensive health status for API health endpoint."""
    health_status = container.health_check()
    healthy_services = [
        name for name, status in health_status.items() if status.get("healthy", False)
    ]
    total_services = len(health_status)

    return {
        "status": "healthy" if len(healthy_services) == total_services else "degraded",
        "services": health_status,
        "summary": {
            "healthy_count": len(healthy_services),
            "total_count": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": [
                name for name in health_status.keys() if name not in healthy_services
            ],
        },
    }
