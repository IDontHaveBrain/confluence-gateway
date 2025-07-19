"""Simple service factory functions to replace complex ServiceRegistry pattern."""

import logging

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.adapters.embedding.factory import get_embedding_provider
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.core.config import (
    confluence_config,
    embedding_config,
    generation_config,
    get_development_context,
    indexing_config,
    search_config,
)
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing_service import IndexingService
from confluence_gateway.services.search import SearchService
from confluence_gateway.services.search_modules.search_validator import SearchValidator

logger = logging.getLogger(__name__)


def create_confluence_client() -> ConfluenceClient | None:
    """Create Confluence client with development mode support."""
    dev_context = get_development_context()
    if dev_context.enabled:
        dev_context.log_stub("Confluence client")
        return None
    return ConfluenceClient(confluence_config) if confluence_config else None


def create_embedding_provider() -> EmbeddingProvider | None:
    """Create embedding provider with development mode support."""
    if not embedding_config or embedding_config.provider == "none":
        return None
    return get_embedding_provider(embedding_config)


def create_vector_db_adapter() -> VectorDBAdapter | None:
    """Create vector database adapter."""
    return get_vector_db_adapter()


def create_embedding_service(
    embedding_provider: EmbeddingProvider | None,
) -> EmbeddingService | None:
    """Create embedding service with provider dependency."""
    if not embedding_provider:
        return None
    return EmbeddingService(embedding_provider)


def create_search_service(
    confluence_client: ConfluenceClient | None,
    vector_db_adapter: VectorDBAdapter | None,
    embedding_service: EmbeddingService | None,
    indexing_service: IndexingService | None,
) -> SearchService | None:
    """Create search service with required dependencies."""
    if not confluence_client:
        return None
    search_validator = SearchValidator()
    return SearchService(
        client=confluence_client,
        indexing_service=indexing_service,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
        search_validator=search_validator,
    )


def create_indexing_service(
    confluence_client: ConfluenceClient | None,
    vector_db_adapter: VectorDBAdapter | None,
    embedding_service: EmbeddingService | None,
) -> IndexingService | None:
    """Create indexing service with required dependencies."""
    if not confluence_client or not vector_db_adapter:
        return None
    return IndexingService(
        confluence_client=confluence_client,
        indexing_config=indexing_config,
        search_config=search_config,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )


def create_generation_service(
    search_service: SearchService | None,
) -> GenerationService | None:
    """Create generation service with search service dependency."""
    if not search_service or not generation_config:
        return None
    return GenerationService(search_service=search_service, config=generation_config)
