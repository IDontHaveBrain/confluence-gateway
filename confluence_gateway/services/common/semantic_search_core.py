"""Core semantic search utilities to eliminate code duplication."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.exceptions import (
    EmbeddingCompatibilityError,
    EmbeddingError,
    SemanticSearchError,
)

if TYPE_CHECKING:
    from confluence_gateway.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticSearchCore:
    """Core utilities for semantic search operations to eliminate code duplication."""

    @staticmethod
    def validate_semantic_search_compatibility(
        embedding_service: EmbeddingService,
        vector_db_adapter: VectorDBAdapter,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """Validate embedding compatibility with vector database for search operations.

        Args:
            embedding_service: Service for text embedding operations
            vector_db_adapter: Adapter for vector database operations
            logger_instance: Optional logger instance to use

        Raises:
            SemanticSearchError: If embedding compatibility validation fails
        """
        log = logger_instance or logger

        try:
            embedding_service.validate_compatibility_with_vector_db(
                vector_db_adapter, operation_type="search"
            )
        except EmbeddingCompatibilityError as e:
            log.error(
                f"Embedding compatibility validation failed for semantic search: {e}"
            )
            raise SemanticSearchError(
                f"Cannot perform semantic search due to embedding model incompatibility: {e}"
            ) from e

    @staticmethod
    def generate_query_embedding(
        query: str,
        embedding_service: EmbeddingService,
        logger_instance: logging.Logger | None = None,
    ) -> list[float]:
        """Generate embedding for search query with comprehensive error handling.

        Args:
            query: Search query text
            embedding_service: Service for text embedding operations
            logger_instance: Optional logger instance to use

        Returns:
            Query embedding as list of floats

        Raises:
            SemanticSearchError: If embedding generation fails
        """
        log = logger_instance or logger

        try:
            log.debug(f"Generating embedding for query: '{query}'")
            query_embedding = embedding_service.embed_text(query)
            if not query_embedding:
                log.error(
                    f"Embedding service returned an empty embedding for query: '{query}'"
                )
                raise SemanticSearchError("Failed to generate a valid query embedding.")
            log.debug("Query embedding generated successfully.")
            return query_embedding

        except EmbeddingError as e:
            log.error(f"Embedding failed for query '{query}': {e}", exc_info=True)
            raise SemanticSearchError(
                f"Failed to generate embedding for the query: {e}"
            ) from e
        except Exception as e:
            log.error(f"Unexpected error during query embedding: {e}", exc_info=True)
            raise SemanticSearchError(
                f"An unexpected error occurred during query embedding: {e}"
            ) from e

    @staticmethod
    def execute_vector_search(
        query_embedding: list[float],
        vector_db_adapter: VectorDBAdapter,
        top_k: int,
        filters: dict[str, Any] | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> list[VectorSearchResultItem]:
        """Execute vector database search with comprehensive error handling.

        Args:
            query_embedding: Query embedding vector
            vector_db_adapter: Adapter for vector database operations
            top_k: Maximum number of results to return
            filters: Optional filters for the search
            logger_instance: Optional logger instance to use

        Returns:
            List of vector search result items

        Raises:
            SemanticSearchError: If vector database search fails
        """
        log = logger_instance or logger

        try:
            log.debug(
                f"Searching vector database with top_k={top_k} and filters={filters}"
            )
            results: list[VectorSearchResultItem] = vector_db_adapter.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            log.debug(f"Vector database search returned {len(results)} results.")
            return results

        except Exception as e:
            log.error(f"Vector database search failed: {e}", exc_info=True)
            raise SemanticSearchError(
                f"Semantic search failed during vector database query: {e}"
            ) from e

    @staticmethod
    def perform_complete_semantic_search(
        query: str,
        embedding_service: EmbeddingService,
        vector_db_adapter: VectorDBAdapter,
        top_k: int,
        filters: dict[str, Any] | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> list[VectorSearchResultItem]:
        """Perform complete semantic search operation with all validations and error handling.

        This method combines compatibility validation, embedding generation, and vector search
        into a single operation to further reduce duplication.

        Args:
            query: Search query text
            embedding_service: Service for text embedding operations
            vector_db_adapter: Adapter for vector database operations
            top_k: Maximum number of results to return
            filters: Optional filters for the search
            logger_instance: Optional logger instance to use

        Returns:
            List of vector search result items

        Raises:
            SemanticSearchError: If any step of the semantic search fails
        """
        log = logger_instance or logger

        # Step 1: Validate compatibility
        SemanticSearchCore.validate_semantic_search_compatibility(
            embedding_service, vector_db_adapter, log
        )

        # Step 2: Generate query embedding
        query_embedding = SemanticSearchCore.generate_query_embedding(
            query, embedding_service, log
        )

        # Step 3: Execute vector search
        return SemanticSearchCore.execute_vector_search(
            query_embedding, vector_db_adapter, top_k, filters, log
        )
