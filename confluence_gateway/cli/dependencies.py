import logging
from typing import Optional

import typer
from rich import print as rich_print

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.core.config import (
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


def _get_confluence_client() -> ConfluenceClient:
    if not confluence_config:
        rich_print(
            "[bold red]Error:[/bold red] Confluence configuration (CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN) is missing or invalid."
        )
        raise typer.Exit(code=1)
    try:
        return ConfluenceClient(config=confluence_config)
    except Exception as e:
        rich_print(
            f"[bold red]Error:[/bold red] Failed to initialize Confluence client: {e}"
        )
        logger.error("Failed to initialize Confluence client", exc_info=True)
        raise typer.Exit(code=1)


def _get_embedding_provider() -> Optional[EmbeddingProvider]:
    try:
        return get_embedding_provider(embedding_config)
    except Exception as e:
        rich_print(
            f"[bold red]Error:[/bold red] Unexpected error getting embedding provider: {e}"
        )
        logger.error("Unexpected error getting embedding provider", exc_info=True)
        return None


def _get_embedding_service() -> EmbeddingService:
    provider = _get_embedding_provider()
    return EmbeddingService(provider=provider)


def _get_vector_db_adapter() -> Optional[VectorDBAdapter]:
    try:
        return get_vector_db_adapter()
    except Exception as e:
        rich_print(
            f"[bold red]Error:[/bold red] Unexpected error getting vector DB adapter: {e}"
        )
        logger.error("Unexpected error getting vector DB adapter", exc_info=True)
        return None


def _get_search_service() -> SearchService:
    client = _get_confluence_client()
    embedding_service = _get_embedding_service()
    vector_db_adapter = _get_vector_db_adapter()
    return SearchService(
        client=client,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )


def _get_indexing_service() -> Optional[IndexingService]:
    if not vector_db_config or vector_db_config.type == "none":
        rich_print(
            "[yellow]Warning:[/yellow] Vector DB is not configured (VECTOR_DB_TYPE=none). Indexing service is unavailable."
        )
        return None

    client = _get_confluence_client()
    embedding_service = _get_embedding_service()

    if not embedding_service.provider:
        rich_print(
            "[yellow]Warning:[/yellow] Embedding provider is not configured or failed to initialize. Indexing service is unavailable as it requires embeddings."
        )
        return None

    try:
        service = IndexingService(
            confluence_client=client,
            indexing_config=indexing_config,
            search_config=search_config,
            embedding_service=embedding_service,
        )
        if not service.vector_db_adapter:
            rich_print(
                "[bold red]Error:[/bold red] IndexingService initialized, but Vector DB Adapter failed to set up internally."
            )
            logger.error("IndexingService vector_db_adapter is None after init.")
            return None
        return service
    except Exception as e:
        rich_print(
            f"[bold red]Error:[/bold red] Failed to initialize Indexing service: {e}"
        )
        logger.error("Failed to initialize Indexing service", exc_info=True)
        return None


def _get_generation_service() -> GenerationService:
    if not generation_config or not generation_config.enable:
        rich_print(
            "[bold red]Error:[/bold red] RAG Generation is disabled in configuration (GENERATION_ENABLE=False)."
        )
        raise typer.Exit(code=1)

    search_service = _get_search_service()

    try:
        service = GenerationService(
            search_service=search_service, config=generation_config
        )
        if not service.config or not service.config.enable:
            rich_print(
                "[bold red]Error:[/bold red] Generation service could not be properly initialized (e.g., missing dependencies). Check logs."
            )
            logger.error("GenerationService config is None or disabled after init.")
            raise typer.Exit(code=1)
        return service
    except Exception as e:
        rich_print(
            f"[bold red]Error:[/bold red] Failed to initialize Generation service: {e}"
        )
        logger.error("Failed to initialize Generation service", exc_info=True)
        raise typer.Exit(code=1)
