"""Hybrid search strategy implementation."""

import logging
import time

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ContentType,
    SearchResult,
)
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.core.config import search_config
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    SearchParameterError,
    SemanticSearchError,
)
from confluence_gateway.services.common.semantic_search_core import SemanticSearchCore
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.ranking import reciprocal_rank_fusion
from confluence_gateway.services.search_modules.search_validator import SearchValidator

logger = logging.getLogger(__name__)


class HybridSearchStrategy:
    """Strategy for performing hybrid search combining keyword and semantic search."""

    def __init__(
        self,
        client: ConfluenceClient,
        embedding_service: EmbeddingService,
        vector_db_adapter: VectorDBAdapter,
        search_validator: SearchValidator | None = None,
    ):
        """Initialize the hybrid search strategy.

        Args:
            client: Confluence client for API operations
            embedding_service: Service for text embedding operations
            vector_db_adapter: Adapter for vector database operations
            search_validator: Optional validator for search parameters
        """
        self.client = client
        self.embedding_service = embedding_service
        self.vector_db_adapter = vector_db_adapter
        self.search_validator = search_validator or SearchValidator()

    def execute_keyword_search(
        self,
        query: str,
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
        expand: list[str] | None = None,
    ) -> tuple[SearchResult | None, dict[str, int]]:
        """Execute keyword search and return results with rankings.

        Args:
            query: Search query text
            content_type: Optional content type filter
            space_key: Optional space key filter
            include_archived: Whether to include archived content
            expand: List of fields to expand in results

        Returns:
            Tuple of (search_result, keyword_ranks_dict)

        Raises:
            Exception: If keyword search fails
        """
        search_result_kw: SearchResult | None = None
        keyword_ranks: dict[str, int] = {}

        try:
            logger.info(
                f"Hybrid Search: Fetching keyword results (limit={search_config.hybrid_keyword_fetch_limit})..."
            )
            search_result_kw = self.client.search(
                query=query,
                content_type=content_type,
                space_key=space_key,
                include_archived=include_archived,
                limit=search_config.hybrid_keyword_fetch_limit,
                start=0,
                expand=expand,
                get_all_results=False,
            )
            keyword_ranks = {
                doc.id: rank
                for rank, doc in enumerate(
                    search_result_kw.results if search_result_kw else [], 1
                )
            }
            logger.info(f"Hybrid Search: Fetched {len(keyword_ranks)} keyword results.")

        except Exception as e:
            logger.error(f"Hybrid Search: Keyword search failed: {e}", exc_info=True)
            raise

        return search_result_kw, keyword_ranks

    def execute_semantic_search(
        self,
        query: str,
        space_key: str | None = None,
        content_type: ContentType | str | None = None,
    ) -> tuple[dict[str, int], list[tuple[str, float]]]:
        """Execute semantic search and return results with rankings.

        Args:
            query: Search query text
            space_key: Optional space key filter
            content_type: Optional content type filter

        Returns:
            Tuple of (semantic_ranks_dict, semantic_results_list)

        Raises:
            SemanticSearchError: If semantic search fails
        """
        semantic_ranks: dict[str, int] = {}
        semantic_results_list: list[tuple[str, float]] = []

        try:
            logger.info(
                f"Hybrid Search: Fetching semantic results (limit={search_config.hybrid_semantic_fetch_limit})..."
            )
            semantic_filters = {}
            if space_key:
                semantic_filters["space_key"] = space_key
            if content_type:
                doc_type_key = "document_type"
                semantic_filters[doc_type_key] = (
                    str(content_type.value)
                    if isinstance(content_type, ContentType)
                    else content_type
                )

            # Validate semantic search parameters using centralized validator
            validated_params = (
                self.search_validator.validate_semantic_search_parameters(
                    query=query, top_k=search_config.hybrid_semantic_fetch_limit
                )
            )
            sanitized_query = validated_params["query"]

            # Use SemanticSearchCore to perform the semantic search operation
            semantic_results_raw = SemanticSearchCore.perform_complete_semantic_search(
                query=sanitized_query,
                embedding_service=self.embedding_service,
                vector_db_adapter=self.vector_db_adapter,
                top_k=search_config.hybrid_semantic_fetch_limit,
                filters=semantic_filters,
                logger_instance=logger,
            )

            semantic_page_scores: dict[str, float] = {}
            for item in semantic_results_raw:
                original_content_id = item.metadata.get("original_content_id")
                if original_content_id:
                    current_score = semantic_page_scores.get(original_content_id, -1.0)
                    semantic_page_scores[original_content_id] = max(
                        current_score, item.score
                    )

            sorted_semantic_pages = sorted(
                semantic_page_scores.items(), key=lambda item: item[1], reverse=True
            )
            semantic_ranks = {
                page_id: rank
                for rank, (page_id, _) in enumerate(sorted_semantic_pages, 1)
            }
            semantic_results_list = list(sorted_semantic_pages)
            logger.info(
                f"Hybrid Search: Processed {len(semantic_ranks)} unique semantic results."
            )

        except SemanticSearchError as e:
            logger.error(f"Hybrid Search: Semantic search failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"Hybrid Search: Unexpected error during semantic search part: {e}",
                exc_info=True,
            )
            raise

        return semantic_ranks, semantic_results_list

    def apply_rrf_ranking(
        self,
        keyword_result_ids: list[str],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Apply Reciprocal Rank Fusion to combine keyword and semantic results.

        Args:
            keyword_result_ids: List of document IDs from keyword search
            semantic_results: List of (document_id, score) tuples from semantic search

        Returns:
            List of (document_id, rrf_score) tuples sorted by RRF score

        Raises:
            SearchParameterError: If RRF input validation fails
            RuntimeError: If RRF processing fails
        """
        rrf_results: list[tuple[str, float]] = []

        try:
            logger.info("Hybrid Search: Performing Reciprocal Rank Fusion...")

            rrf_results = reciprocal_rank_fusion(
                keyword_result_ids=keyword_result_ids,
                semantic_results=semantic_results,
                k=search_config.hybrid_rrf_k,
            )
            logger.info(
                f"Hybrid Search: RRF completed, {len(rrf_results)} total ranked results."
            )

        except ValueError as e:
            logger.error(
                f"Hybrid Search: RRF input validation failed: {e}", exc_info=True
            )
            raise SearchParameterError(f"RRF failed: {e}") from e
        except Exception as e:
            logger.error(f"Hybrid Search: RRF failed: {e}", exc_info=True)
            raise RuntimeError(f"Hybrid search failed during RRF: {e}") from e

        return rrf_results

    def fetch_final_documents(
        self,
        rrf_results: list[tuple[str, float]],
        cached_docs: dict[str, ConfluencePage],
        start: int,
        limit: int,
        expand: list[str] | None = None,
    ) -> list[ConfluencePage]:
        """Fetch final documents based on RRF results with pagination.

        Args:
            rrf_results: List of (document_id, score) tuples from RRF
            cached_docs: Dictionary of already fetched documents
            start: Start position for pagination
            limit: Maximum number of results to return
            expand: List of fields to expand in results

        Returns:
            List of ConfluencePage objects in RRF order
        """
        paginated_rrf_results = rrf_results[start : start + limit]
        final_doc_ids = [doc_id for doc_id, score in paginated_rrf_results]

        logger.info(
            f"Hybrid Search: Applied pagination (start={start}, limit={limit}), {len(final_doc_ids)} final IDs."
        )

        final_page_objects: list[ConfluencePage] = []
        fetched_docs_cache = dict(cached_docs)

        for doc_id in final_doc_ids:
            if doc_id in fetched_docs_cache:
                final_page_objects.append(fetched_docs_cache[doc_id])
            else:
                try:
                    logger.debug(
                        f"Hybrid Search: Fetching details for missing ID {doc_id}..."
                    )
                    page = self.client.get_page(doc_id, expand=expand)
                    final_page_objects.append(page)
                    fetched_docs_cache[doc_id] = page
                except ConfluenceAPIError as e:
                    if e.status_code == 404:
                        logger.warning(
                            f"Hybrid Search: Document ID {doc_id} (from RRF) not found in Confluence. Skipping."
                        )
                    else:
                        logger.error(
                            f"Hybrid Search: Failed to fetch details for document ID {doc_id}: {e}",
                            exc_info=True,
                        )
                except Exception as e:
                    logger.error(
                        f"Hybrid Search: Unexpected error fetching details for document ID {doc_id}: {e}",
                        exc_info=True,
                    )

        logger.info(
            f"Hybrid Search: Successfully prepared {len(final_page_objects)} final document objects."
        )

        return final_page_objects

    def search_hybrid(
        self,
        text: str,
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
    ) -> SearchResult:
        """Execute hybrid search combining keyword and semantic search.

        Args:
            text: Search query text (sanitized)
            content_type: Optional content type filter
            space_key: Optional space key filter
            include_archived: Whether to include archived content
            limit: Maximum number of results to return
            start: Start position for pagination
            expand: List of fields to expand in results

        Returns:
            SearchResult object with hybrid search results

        Raises:
            SemanticSearchError: If hybrid search is disabled or services unavailable
            EmbeddingCompatibilityError: If embedding compatibility check fails
        """
        if not search_config.hybrid_search_enabled:
            raise SemanticSearchError("Hybrid search is disabled in the configuration.")

        if not self.embedding_service or not self.vector_db_adapter:
            raise SemanticSearchError(
                "Hybrid search requires EmbeddingService and VectorDBAdapter to be configured."
            )

        actual_expand = expand if expand is not None else search_config.default_expand

        start_time = time.time()

        # Phase 1: Execute keyword search
        search_result_kw, keyword_ranks = self.execute_keyword_search(
            query=text,
            content_type=content_type,
            space_key=space_key,
            include_archived=include_archived,
            expand=actual_expand,
        )

        # Phase 2: Execute semantic search
        semantic_ranks, semantic_results_list = self.execute_semantic_search(
            query=text,
            space_key=space_key,
            content_type=content_type,
        )

        # Phase 3: Apply RRF ranking
        keyword_result_ids_list = (
            [doc.id for doc in search_result_kw.results] if search_result_kw else []
        )
        rrf_results = self.apply_rrf_ranking(
            keyword_result_ids=keyword_result_ids_list,
            semantic_results=semantic_results_list,
        )

        # Phase 4: Apply pagination and fetch final documents
        total_hybrid_results = len(rrf_results)
        final_start = start or 0
        final_limit = limit or search_config.default_limit

        cached_docs = {
            doc.id: doc
            for doc in (search_result_kw.results if search_result_kw else [])
        }

        final_page_objects = self.fetch_final_documents(
            rrf_results=rrf_results,
            cached_docs=cached_docs,
            start=final_start,
            limit=final_limit,
            expand=actual_expand,
        )

        # Phase 5: Build final search result
        final_search_result_obj = SearchResult(
            results=final_page_objects,
            total_size=total_hybrid_results,
            start=final_start,
            limit=final_limit,
        )

        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Hybrid search completed in {execution_time_ms:.2f} ms")

        return final_search_result_obj
