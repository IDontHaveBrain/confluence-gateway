"""CLI dependency management using ServiceContainer infrastructure."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.cli.common import print_status
from confluence_gateway.core.config import get_development_context
from confluence_gateway.core.service_container import CLIErrorStrategy, ServiceContainer
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing_service import IndexingService
from confluence_gateway.services.search import SearchService


# Stub implementations for development mode
class StubConfluenceClient:
    """Stub implementation of ConfluenceClient for development mode."""

    def list_spaces_paginated(
        self, start: int = 0, limit: int = 25, **kwargs: Any
    ) -> tuple[list[Any], int]:
        """Return stub spaces data as tuple (spaces, total_count)."""
        # Create mock space objects that match expected structure
        mock_spaces = [
            type(
                "MockSpace",
                (),
                {
                    "id": "12345",
                    "key": "DEV",
                    "name": "Development Space",
                    "title": "Development Space",
                    "type": type("SpaceType", (), {"value": "global"})(),
                },
            )()
        ]
        return mock_spaces, 1  # (spaces, total_count)

    def list_all_spaces(self, **kwargs: Any) -> list[Any]:
        """Return all stub spaces data."""
        # Create mock space objects that match expected structure
        mock_spaces = [
            type(
                "MockSpace",
                (),
                {
                    "id": "12345",
                    "key": "DEV",
                    "name": "Development Space",
                    "title": "Development Space",
                    "type": type("SpaceType", (), {"value": "global"})(),
                },
            )(),
            type(
                "MockSpace",
                (),
                {
                    "id": "67890",
                    "key": "TEST",
                    "name": "Test Space",
                    "title": "Test Space",
                    "type": type("SpaceType", (), {"value": "global"})(),
                },
            )(),
        ]
        return mock_spaces

    def get_space(self, space_key: str, **kwargs: Any) -> Any:
        """Return stub space data as space object."""
        return type(
            "MockSpace",
            (),
            {
                "id": "12345",
                "key": space_key,
                "name": f"{space_key} Space",
                "title": f"{space_key} Space",
                "type": type("SpaceType", (), {"value": "global"})(),
                "description_text": f"Stub description for {space_key}",
                "created_at": None,
                "updated_at": None,
            },
        )()

    def search_by_cql(self, cql: str, **kwargs: Any) -> dict[str, Any]:
        """Return stub CQL search results."""
        return {
            "results": [
                {
                    "id": "page123",
                    "title": f"Stub result for: {cql}",
                    "type": "page",
                    "space": {"key": "DEV", "name": "Development Space"},
                    "body": {"view": {"value": f"<p>Stub content for CQL: {cql}</p>"}},
                    "_links": {"webui": "/pages/page123"},
                }
            ],
            "start": 0,
            "limit": 25,
            "size": 1,
            "searchDuration": 50,
            "_links": {},
        }

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Return stub search results."""
        return {
            "results": [
                {
                    "id": "page123",
                    "title": f"Stub result for: {query}",
                    "type": "page",
                    "space": {"key": "DEV", "name": "Development Space"},
                    "body": {
                        "view": {"value": f"<p>Stub content for query: {query}</p>"}
                    },
                    "_links": {"webui": "/pages/page123"},
                }
            ],
            "start": 0,
            "limit": 25,
            "size": 1,
            "searchDuration": 50,
            "_links": {},
        }

    def extract_content_fields(self, item: Any) -> dict[str, Any]:
        """Extract content fields from a search result item."""
        return {
            "title": getattr(item, "title", "Stub Title"),
            "content": getattr(item, "content", "Stub content"),
            "excerpt": getattr(item, "excerpt", "Stub excerpt"),
            "url": getattr(item, "url", "/stub/url"),
            "space": getattr(
                item, "space", {"key": "DEV", "name": "Development Space"}
            ),
            "type": getattr(item, "type", "page"),
        }


class StubSearchService:
    """Stub implementation of SearchService for development mode."""

    def __init__(self) -> None:
        """Initialize stub search service with required attributes."""
        # Provide a reference to the stub confluence client
        self.client = StubConfluenceClient()

    def search_by_text(self, text: str, **kwargs: Any) -> Any:
        """Return stub text search results."""
        # Return an enhanced search result object with the expected structure
        return type(
            "EnhancedSearchResult",
            (),
            {
                "results": type(
                    "SearchResults",
                    (),
                    {
                        "results": [
                            type(
                                "SearchResult",
                                (),
                                {
                                    "id": "page123",
                                    "title": f"Text search result for: {text}",
                                    "type": "page",
                                    "space": type(
                                        "Space",
                                        (),
                                        {"key": "DEV", "name": "Development Space"},
                                    )(),
                                    "url": "/pages/page123",
                                    "content": f"Stub content for text search: {text}",
                                    "excerpt": f"...{text}...",
                                    "score": 0.95,
                                },
                            )()
                        ],
                        "start": 0,
                        "limit": 25,
                        "size": 1,
                    },
                )(),
                "took_ms": 100,
                "statistics": type(
                    "Statistics",
                    (),
                    {
                        "total_count": 1,
                        "processed_count": 1,
                        "filtered_count": 1,
                        "total_results": 1,
                        "execution_time_ms": 100,
                    },
                )(),
            },
        )()

    def search_semantic(self, query: str, **kwargs: Any) -> tuple[list[Any], int]:
        """Return stub semantic search results."""
        results = [
            type(
                "VectorSearchResult",
                (),
                {
                    "id": "page124",
                    "score": 0.89,
                    "text": f"Stub semantic content for: {query}",
                    "metadata": {
                        "title": f"Semantic result for: {query}",
                        "space_key": "DEV",
                        "space_name": "Development Space",
                        "type": "page",
                        "url": "/pages/page124",
                    },
                },
            )()
        ]
        return results, 150

    def search_by_cql(self, cql: str, **kwargs: Any) -> Any:
        """Return stub CQL search results with enhanced result format."""
        # Return an enhanced search result object with the expected structure like search_by_text
        return type(
            "EnhancedSearchResult",
            (),
            {
                "results": type(
                    "SearchResults",
                    (),
                    {
                        "results": [
                            type(
                                "SearchResult",
                                (),
                                {
                                    "id": "page125",
                                    "title": f"CQL result for: {cql}",
                                    "type": "page",
                                    "space": type(
                                        "Space",
                                        (),
                                        {"key": "DEV", "name": "Development Space"},
                                    )(),
                                    "url": "/pages/page125",
                                    "content": f"Stub CQL content for: {cql}",
                                    "excerpt": f"...{cql}...",
                                    "score": 0.92,
                                },
                            )()
                        ],
                        "start": 0,
                        "limit": 25,
                        "size": 1,
                    },
                )(),
                "took_ms": 80,
                "statistics": type(
                    "Statistics",
                    (),
                    {
                        "total_count": 1,
                        "processed_count": 1,
                        "filtered_count": 1,
                        "total_results": 1,
                        "execution_time_ms": 80,
                    },
                )(),
            },
        )()

    def search_hybrid(self, text: str, **kwargs: Any) -> Any:
        """Return stub hybrid search results with enhanced result format."""
        # Return an enhanced search result object with the expected structure like search_by_text
        return type(
            "EnhancedSearchResult",
            (),
            {
                "results": type(
                    "SearchResults",
                    (),
                    {
                        "results": [
                            type(
                                "SearchResult",
                                (),
                                {
                                    "id": "page126",
                                    "title": f"Hybrid result for: {text}",
                                    "type": "page",
                                    "space": type(
                                        "Space",
                                        (),
                                        {"key": "DEV", "name": "Development Space"},
                                    )(),
                                    "url": "/pages/page126",
                                    "content": f"Stub hybrid content for: {text}",
                                    "excerpt": f"...{text}...",
                                    "score": 0.97,
                                },
                            )()
                        ],
                        "start": 0,
                        "limit": 25,
                        "size": 1,
                    },
                )(),
                "took_ms": 200,
                "statistics": type(
                    "Statistics",
                    (),
                    {
                        "total_count": 1,
                        "processed_count": 1,
                        "filtered_count": 1,
                        "total_results": 1,
                        "execution_time_ms": 200,
                    },
                )(),
            },
        )()


class StubGenerationService:
    """Stub implementation of GenerationService for development mode."""

    async def generate_answer(
        self,
        query: str,
        top_k_retrieval: int = 5,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return stub generation response as tuple (answer, retrieved_results)."""
        answer = f"This is a stub answer for the question: {query}. In development mode, this would normally connect to your configured AI model to generate a comprehensive response based on Confluence content."

        retrieved_results = [
            type(
                "VectorSearchResult",
                (),
                {
                    "metadata": {
                        "title": f"Stub source for: {query}",
                        "url": "/pages/source123",
                        "space_key": "DEV",
                        "space_name": "Development Space",
                        "excerpt": f"Relevant content related to: {query}",
                    },
                    "score": 0.95,
                    "text": f"Relevant content related to: {query}",
                    "id": "source123",
                },
            )()
        ]

        return answer, retrieved_results


class StubIndexingService:
    """Stub implementation of IndexingService for development mode."""

    def __init__(self) -> None:
        """Initialize stub indexing service with status."""
        self._status = {
            "status": "idle",
            "last_run_start_time": None,
            "last_run_end_time": None,
            "last_error_message": None,
        }

    @property
    def status(self) -> dict[str, Any]:
        """Return current indexing status."""
        return self._status

    async def run_indexing(
        self,
        space_keys: list[str] | None = None,
        index_all: bool = False,
        **kwargs: Any,
    ) -> None:
        """Mock indexing operation for dev mode."""
        import asyncio
        from datetime import datetime, timezone

        self._status["status"] = "running"
        self._status["last_run_start_time"] = datetime.now(timezone.utc).isoformat()

        # Simulate quick indexing
        await asyncio.sleep(0.1)

        self._status["status"] = "success"
        self._status["last_run_end_time"] = datetime.now(timezone.utc).isoformat()
        self._status["last_error_message"] = None

    def _run_indexing_sync(
        self,
        space_keys: list[str] | None = None,
        index_all: bool = False,
        **kwargs: Any,
    ) -> None:
        """Synchronous version of run_indexing for CLI."""
        import time
        from datetime import datetime, timezone

        self._status["status"] = "running"
        self._status["last_run_start_time"] = datetime.now(timezone.utc).isoformat()

        # Simulate quick indexing
        time.sleep(0.1)

        self._status["status"] = "success"
        self._status["last_run_end_time"] = datetime.now(timezone.utc).isoformat()
        self._status["last_error_message"] = None

    def trigger_indexing(
        self, spaces: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Return stub indexing trigger response."""
        target_spaces = spaces or ["ALL"]
        return {
            "status": "started",
            "message": f"Indexing triggered for spaces: {', '.join(target_spaces)} (stub mode)",
            "job_id": "stub-job-123",
            "estimated_duration": "5-10 minutes",
            "timestamp": datetime.now().isoformat(),
        }

    def get_indexing_status(self, **kwargs: Any) -> dict[str, Any]:
        """Return stub indexing status."""
        return {
            "status": "idle",
            "message": "No indexing job currently running (stub mode)",
            "last_run": {
                "started_at": (datetime.now()).isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_ms": 30000,
                "indexed_pages": 42,
                "status": "completed",
            },
        }


def _check_dev_mode_feature(feature_name: str) -> None:
    """Check if a feature is available when dev mode is enabled."""
    dev_context = get_development_context()
    if dev_context.enabled:
        print_status(
            f"Notice: {feature_name} is running in DEV MODE - using stub implementation for faster development iteration. "
            f"To access full functionality, run without CONFLUENCE_GATEWAY_DEV_MODE environment variable.",
            "warning",
        )


# Global ServiceContainer instance optimized for CLI performance
_container: ServiceContainer | None = None
_cli_error_strategy = CLIErrorStrategy()


def _get_container() -> ServiceContainer:
    """Get the singleton ServiceContainer instance optimized for CLI."""
    global _container
    if _container is None:
        # Use non-thread-safe mode for maximum CLI performance
        _container = ServiceContainer(thread_safe=False)
        _setup_service_factories(_container)
    return _container


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
    def confluence_client_factory() -> ConfluenceClient | StubConfluenceClient:
        dev_context = get_development_context()
        if dev_context.enabled:
            dev_context.log_stub("Confluence client")
            return StubConfluenceClient()
        return ConfluenceClient(confluence_config)

    # Search service factory
    def search_service_factory() -> SearchService | StubSearchService | None:
        dev_context = get_development_context()
        if dev_context.enabled:
            dev_context.log_stub("Search service")
            return StubSearchService()

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
    def indexing_service_factory() -> IndexingService | StubIndexingService | None:
        dev_context = get_development_context()

        if dev_context.enabled:
            dev_context.log_stub("Indexing service")
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
    def generation_service_factory() -> (
        GenerationService | StubGenerationService | None
    ):
        dev_context = get_development_context()

        if dev_context.enabled:
            dev_context.log_stub("Generation service")
            return StubGenerationService()

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


def _get_confluence_client() -> ConfluenceClient | StubConfluenceClient:
    """Get Confluence client with typer.Exit error handling."""
    container = _get_container()
    return cast(
        ConfluenceClient | StubConfluenceClient,
        container.get_service("confluence_client", error_strategy=_cli_error_strategy),
    )


def _get_search_service() -> SearchService | StubSearchService:
    """Get search service with typer.Exit error handling."""
    container = _get_container()
    return cast(
        SearchService | StubSearchService,
        container.get_service("search_service", error_strategy=_cli_error_strategy),
    )


def _get_indexing_service() -> IndexingService | StubIndexingService | None:
    """Get indexing service with safe error handling."""
    container = _get_container()
    try:
        return cast(
            IndexingService | StubIndexingService | None,
            container.get_service("indexing_service"),
        )
    except Exception:
        return None


def _get_generation_service() -> GenerationService | StubGenerationService:
    """Get generation service with typer.Exit error handling."""
    container = _get_container()
    dev_context = get_development_context()
    if dev_context.enabled:
        _check_dev_mode_feature("RAG Generation")
    return cast(
        GenerationService | StubGenerationService,
        container.get_service("generation_service", error_strategy=_cli_error_strategy),
    )
