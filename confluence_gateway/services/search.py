import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Union, cast

from pydantic import BaseModel, Field

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ContentType,
    SearchResult,
)
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.config import get_development_context, search_config
from confluence_gateway.core.exceptions import (
    SemanticSearchError,
)
from confluence_gateway.services.common.initialization_logger import (
    InitializationLogger,
)
from confluence_gateway.services.common.semantic_search_core import SemanticSearchCore
from confluence_gateway.services.common.validation_utils import ValidationUtils
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.indexing_service import IndexingService
from confluence_gateway.services.search_modules.search_validator import SearchValidator
from confluence_gateway.services.search_modules.strategies.hybrid_search import (
    HybridSearchStrategy,
)

logger = logging.getLogger(__name__)

SearchResult_T = Union[
    SearchResult, "EnhancedSearchResult", list[VectorSearchResultItem]
]


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    TITLE = "title"
    CREATED = "created_at"
    UPDATED = "updated_at"
    RELEVANCE = "score"
    SPACE = "space_key"


class SearchStatistics(BaseModel):
    total_results: int = 0
    filtered_results: int = 0
    total_pages: int = 0
    current_page: int = 0
    execution_time_ms: float = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class EnhancedSearchResult(BaseModel):
    results: SearchResult
    statistics: SearchStatistics
    query: str | None = None
    filters_applied: dict[str, Any] | None = None
    sort_criteria: list[dict[str, str]] | None = None

    def to_standard_result(self) -> SearchResult:
        return self.results


class SearchService:
    def __init__(
        self,
        client: ConfluenceClient | None,
        indexing_service: IndexingService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_db_adapter: VectorDBAdapter | None = None,
        search_validator: SearchValidator | None = None,
    ):
        self.client = client
        self.indexing_service = indexing_service
        self.embedding_service = embedding_service
        self.vector_db_adapter = vector_db_adapter
        self.search_validator = search_validator or SearchValidator()

        # Development mode support
        self.dev_context = get_development_context()
        self.dev_mode = self.dev_context.enabled

        if self.dev_mode:
            self.dev_context.log_stub("SearchService (semantic search)")
            logger.info(
                "SearchService initialized in DEV MODE - stub implementation for semantic/hybrid search."
            )

        # Initialize hybrid search strategy if required components are available
        if self.embedding_service and self.vector_db_adapter and self.client:
            self.hybrid_search_strategy: HybridSearchStrategy | None = (
                HybridSearchStrategy(
                    client=self.client,
                    embedding_service=self.embedding_service,
                    vector_db_adapter=self.vector_db_adapter,
                    search_validator=self.search_validator,
                )
            )
        else:
            self.hybrid_search_strategy = None

        # Log component availability using standardized patterns
        InitializationLogger.log_component_availability(
            "SearchService",
            "HybridSearchStrategy",
            self.hybrid_search_strategy is not None,
            logger,
            impact_message="Hybrid search will be disabled.",
        )

        InitializationLogger.log_component_availability(
            "SearchService",
            "IndexingService",
            self.indexing_service is not None,
            logger,
        )

        InitializationLogger.log_component_availability(
            "SearchService",
            "EmbeddingService",
            self.embedding_service is not None,
            logger,
            impact_message="Semantic search might be disabled.",
        )

        InitializationLogger.log_component_availability(
            "SearchService",
            "VectorDBAdapter",
            self.vector_db_adapter is not None,
            logger,
            impact_message="Semantic search might be disabled.",
        )

    def _prepare_sort_criteria(
        self,
        sort_by: list[SortField | str] | None,
        sort_direction: list[SortDirection | str] | None,
    ) -> list[dict[str, str]] | None:
        if not sort_by:
            return None

        sort_criteria = []
        directions = sort_direction or []

        for i, field in enumerate(sort_by):
            direction = directions[i] if i < len(directions) else SortDirection.ASC

            if isinstance(field, str):
                try:
                    field = SortField(field.lower())
                except ValueError:
                    field = SortField.TITLE

            if isinstance(direction, str):
                try:
                    direction = SortDirection(direction.lower())
                except ValueError:
                    direction = SortDirection.ASC

            sort_criteria.append(
                {"field": str(field.value), "direction": str(direction.value)}
            )

        return sort_criteria

    def _sanitize_keywords(self, keywords: str | list[str]) -> str:
        return self.search_validator.sanitize_keywords(keywords)

    def search_by_text(
        self,
        text: str | list[str],
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
        get_all_results: bool = False,
        max_results: int | None = None,
        min_relevance: float = 0.0,
        top_n: int | None = None,
        sort_by: list[SortField | str] | None = None,
        sort_direction: list[SortDirection | str] | None = None,
        return_enhanced_result: bool = True,
    ) -> SearchResult_T:
        # Validate and normalize search parameters
        validated_params = self.search_validator.validate_and_normalize_search_params(
            limit=limit, start=start
        )
        limit = validated_params["limit"]
        start = validated_params["start"]

        sanitized_text = self.search_validator.sanitize_keywords(text)

        actual_expand = expand
        if actual_expand is None and search_config.default_expand:
            actual_expand = search_config.default_expand

        filters = {
            "content_type": content_type,
            "space_key": space_key,
            "include_archived": include_archived,
            "min_relevance": min_relevance,
            "top_n": top_n,
            "get_all_results": get_all_results,
            "max_results": max_results,
        }

        sort_criteria = self._prepare_sort_criteria(sort_by, sort_direction)

        page_number = ((start or 0) // (limit or 1)) + 1 if (limit or 0) > 0 else 1

        start_time = time.time()

        if self.client is None:
            # Return mock search results for development mode using proper data structures
            from confluence_gateway.adapters.confluence.models import SearchResult

            mock_page = ConfluencePage(
                id="mock-page-1",
                title=f"Mock Search Result for: {sanitized_text}",
                type="page",
                space={"key": "MOCK", "name": "Mock Space"},
                content={
                    "body": {
                        "view": {
                            "value": f"This is a mock search result for query: {sanitized_text}"
                        }
                    }
                },
                url="/mock/page/1",
                _links={"base": "https://mock.atlassian.net", "webui": "/mock/page/1"},
            )

            mock_search_result = SearchResult(
                results=[mock_page], total_size=1, limit=limit or 25, start=start or 0
            )

            if return_enhanced_result:
                statistics = SearchStatistics(
                    total_results=1,
                    filtered_results=1,
                    total_pages=1,
                    current_page=page_number,
                    execution_time_ms=5.0,
                    timestamp=datetime.now(),
                )

                return EnhancedSearchResult(
                    results=mock_search_result,
                    statistics=statistics,
                    query=sanitized_text,
                    filters_applied=filters,
                    sort_criteria=sort_criteria,
                )
            else:
                return cast(SearchResult_T, mock_search_result)

        search_result = self.client.search(
            query=sanitized_text,
            content_type=content_type,
            space_key=space_key,
            include_archived=include_archived,
            limit=limit,
            start=start,
            expand=actual_expand,
            get_all_results=get_all_results,
            max_results=max_results,
        )

        execution_time_ms = (time.time() - start_time) * 1000

        if (min_relevance is not None and min_relevance > 0) or top_n is not None:
            search_result = self._filter_by_relevance(
                search_result, min_score=min_relevance or 0.0, top_n=top_n
            )

        if sort_by:
            search_result = self._sort_results(
                search_result, sort_fields=sort_by, directions=sort_direction
            )

        if return_enhanced_result:
            return self._process_search_result(
                search_result,
                execution_time_ms=execution_time_ms,
                query=sanitized_text,
                filters=filters,
                sort_criteria=sort_criteria,
                current_page=page_number,
            )
        else:
            return cast(SearchResult_T, search_result)

    def _process_search_result(
        self,
        result: SearchResult,
        execution_time_ms: float = 0,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        sort_criteria: list[dict[str, str]] | None = None,
        current_page: int = 1,
    ) -> EnhancedSearchResult:
        items_per_page = result.limit or search_config.default_limit
        total_pages = (
            (result.total_size + items_per_page - 1) // items_per_page
            if items_per_page > 0
            else 0
        )

        statistics = SearchStatistics(
            total_results=result.total_size,
            filtered_results=len(result.results),
            total_pages=total_pages,
            current_page=current_page,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
        )

        enhanced_result = EnhancedSearchResult(
            results=result,
            statistics=statistics,
            query=query,
            filters_applied=filters,
            sort_criteria=sort_criteria,
        )

        return enhanced_result

    def _filter_by_relevance(
        self, results: SearchResult, min_score: float = 0.0, top_n: int | None = None
    ) -> SearchResult:
        filtered_results = list(results.results)

        # Filter by minimum score if specified
        if min_score > 0.0:
            # Note: Confluence text search doesn't provide scores, so we skip score filtering
            # This parameter is kept for API compatibility
            pass

        if top_n is not None and top_n > 0 and len(filtered_results) > top_n:
            filtered_results = filtered_results[:top_n]

        return SearchResult(
            total_size=results.total_size,
            start=results.start,
            limit=results.limit,
            results=filtered_results,
        )

    def _sort_results(
        self,
        results: SearchResult,
        sort_fields: list[SortField | str],
        directions: list[SortDirection | str] | None = None,
    ) -> SearchResult:
        if not results.results or not sort_fields:
            return results

        if directions is None:
            directions = [SortDirection.ASC] * len(sort_fields)
        elif len(directions) < len(sort_fields):
            directions.extend(
                [SortDirection.ASC] * (len(sort_fields) - len(directions))
            )

        normalized_fields = []
        for field in sort_fields:
            if isinstance(field, str):
                try:
                    field = SortField(field.lower())
                except ValueError:
                    field = SortField.TITLE
            normalized_fields.append(field)

        normalized_directions = []
        for direction in directions:
            if isinstance(direction, str):
                try:
                    direction = SortDirection(direction.lower())
                except ValueError:
                    direction = SortDirection.ASC
            normalized_directions.append(direction)

        items_to_sort = list(results.results)

        for sort_idx in range(len(normalized_fields) - 1, -1, -1):
            field = normalized_fields[sort_idx]
            direction = normalized_directions[sort_idx]
            reverse = direction == SortDirection.DESC

            def get_sort_value(item: ConfluencePage) -> Any:
                if field == SortField.TITLE:
                    return item.title or ""
                elif field == SortField.CREATED:
                    return item.created_at or datetime.min
                elif field == SortField.UPDATED:
                    return item.updated_at or datetime.min
                elif field == SortField.SPACE:
                    if hasattr(item, "space_key"):
                        return item.space_key or ""
                    elif isinstance(item.space, dict):
                        return item.space.get("key", "")
                    elif item.space and hasattr(item.space, "key"):
                        return item.space.key or ""
                    else:
                        return ""
                else:
                    return 0

            def key_func(item: ConfluencePage) -> Any:
                return get_sort_value(item)

            items_to_sort.sort(key=key_func, reverse=reverse)

        return SearchResult(
            total_size=results.total_size,
            start=results.start,
            limit=results.limit,
            results=items_to_sort,
        )

    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[VectorSearchResultItem], float]:
        if self.dev_mode:
            logger.info(
                f"DEV MODE: Generating stub semantic search results for query: '{query[:50]}...'"
            )
            # Return empty results list and mock timing
            return [], 50.0

        # Validate required dependencies for semantic search
        dependencies = {
            "EmbeddingService": self.embedding_service,
            "VectorDBAdapter": self.vector_db_adapter,
        }

        for service_name, service in dependencies.items():
            if not ValidationUtils.validate_service_dependency(
                service,
                service_name,
                logger,
                required=True,
                operation_name="semantic search",
            ):
                raise SemanticSearchError(
                    f"Semantic search is not configured: {service_name} is missing."
                )

        # Validate semantic search parameters using centralized validator
        validated_params = self.search_validator.validate_semantic_search_parameters(
            query=query, top_k=top_k
        )
        sanitized_query = validated_params["query"]
        top_k = validated_params["top_k"]
        logger.info(
            f"Performing semantic search for query: '{sanitized_query}', top_k={top_k}, filters={filters}"
        )

        start_time = time.time()

        # Use SemanticSearchCore to perform the complete semantic search operation
        # Assert that services are not None since we validated them above
        assert self.embedding_service is not None, (
            "EmbeddingService should not be None after validation"
        )
        assert self.vector_db_adapter is not None, (
            "VectorDBAdapter should not be None after validation"
        )

        results = SemanticSearchCore.perform_complete_semantic_search(
            query=sanitized_query,
            embedding_service=self.embedding_service,
            vector_db_adapter=self.vector_db_adapter,
            top_k=top_k,
            filters=filters,
            logger_instance=logger,
        )

        took_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Semantic search completed in {took_ms:.2f} ms, found {len(results)} results."
        )

        return results, took_ms

    def search_by_cql(
        self,
        cql: str,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
        get_all_results: bool = False,
        max_results: int | None = None,
        top_n: int | None = None,
        sort_by: list[SortField | str] | None = None,
        sort_direction: list[SortDirection | str] | None = None,
        return_enhanced_result: bool = True,
    ) -> SearchResult_T:
        # Validate and normalize search parameters
        validated_params = self.search_validator.validate_and_normalize_search_params(
            limit=limit, start=start
        )
        limit = validated_params["limit"]
        start = validated_params["start"]

        cql = self.search_validator.validate_cql_query(cql)

        actual_expand = expand
        if actual_expand is None and search_config.default_expand:
            actual_expand = search_config.default_expand

        filters = {
            "cql": cql,
            "top_n": top_n,
            "get_all_results": get_all_results,
            "max_results": max_results,
        }
        sort_criteria = self._prepare_sort_criteria(sort_by, sort_direction)
        page_number = ((start or 0) // (limit or 1)) + 1 if (limit or 0) > 0 else 1

        start_time = time.time()

        if self.client is None:
            # Return mock CQL search results for development mode using proper data structures
            from confluence_gateway.adapters.confluence.models import SearchResult

            mock_page = ConfluencePage(
                id="mock-cql-page-1",
                title=f"Mock CQL Search Result for: {cql}",
                type="page",
                space={"key": "MOCK", "name": "Mock Space"},
                content={
                    "body": {
                        "view": {
                            "value": f"This is a mock CQL search result for query: {cql}"
                        }
                    }
                },
                url="/mock/cql/page/1",
                _links={
                    "base": "https://mock.atlassian.net",
                    "webui": "/mock/cql/page/1",
                },
            )

            mock_search_result = SearchResult(
                results=[mock_page], total_size=1, limit=limit or 25, start=start or 0
            )

            if return_enhanced_result:
                statistics = SearchStatistics(
                    total_results=1,
                    filtered_results=1,
                    total_pages=1,
                    current_page=page_number,
                    execution_time_ms=5.0,
                    timestamp=datetime.now(),
                )

                return EnhancedSearchResult(
                    results=mock_search_result,
                    statistics=statistics,
                    query=cql,
                    filters_applied=filters,
                    sort_criteria=sort_criteria,
                )
            else:
                return cast(SearchResult_T, mock_search_result)

        search_result = self.client.search_by_cql(
            cql=cql,
            limit=limit,
            start=start,
            expand=actual_expand,
            get_all_results=get_all_results,
            max_results=max_results,
        )
        execution_time_ms = (time.time() - start_time) * 1000

        if top_n is not None:
            search_result = self._filter_by_relevance(search_result, top_n=top_n)

        if sort_by:
            search_result = self._sort_results(
                search_result, sort_fields=sort_by, directions=sort_direction
            )

        if return_enhanced_result:
            return self._process_search_result(
                search_result,
                execution_time_ms=execution_time_ms,
                query=f"CQL: {cql}",
                filters=filters,
                sort_criteria=sort_criteria,
                current_page=page_number,
            )
        else:
            return cast(SearchResult_T, search_result)

    def search_hybrid(
        self,
        text: str | list[str],
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
        return_enhanced_result: bool = True,
    ) -> SearchResult_T:
        """Execute hybrid search combining keyword and semantic search using strategy pattern.

        Args:
            text: Search query text or list of keywords
            content_type: Optional content type filter
            space_key: Optional space key filter
            include_archived: Whether to include archived content
            limit: Maximum number of results to return
            start: Start position for pagination
            expand: List of fields to expand in results
            return_enhanced_result: Whether to return enhanced result with statistics

        Returns:
            SearchResult or EnhancedSearchResult based on return_enhanced_result flag

        Raises:
            SemanticSearchError: If hybrid search is disabled or services unavailable
            SearchParameterError: If search parameters are invalid
        """
        # Validate and normalize search parameters
        validated_params = self.search_validator.validate_and_normalize_search_params(
            limit=limit, start=start
        )
        limit = validated_params["limit"]
        start = validated_params["start"]

        start_time = time.time()

        # Sanitize and prepare query
        sanitized_text = self._sanitize_keywords(text)

        # Validate hybrid search strategy availability
        if not ValidationUtils.validate_service_dependency(
            self.hybrid_search_strategy,
            "HybridSearchStrategy",
            logger,
            required=True,
            operation_name="hybrid search",
        ):
            # Check if we're in dev mode - if so, provide fallback to text search
            dev_context = get_development_context()
            if dev_context.enabled:
                # Fallback to text search in dev mode when hybrid search is not available
                return self.search_by_text(
                    text=sanitized_text,
                    content_type=content_type,
                    space_key=space_key,
                    include_archived=include_archived,
                    limit=limit,
                    start=start,
                    expand=expand,
                    return_enhanced_result=return_enhanced_result,
                )
            else:
                raise SemanticSearchError(
                    "Hybrid search is not available. EmbeddingService and VectorDBAdapter are required."
                )

        # Delegate to hybrid search strategy
        # Assert that hybrid_search_strategy is not None since we validated it above
        assert self.hybrid_search_strategy is not None, (
            "HybridSearchStrategy should not be None after validation"
        )

        final_search_result_obj = self.hybrid_search_strategy.search_hybrid(
            text=sanitized_text,
            content_type=content_type,
            space_key=space_key,
            include_archived=include_archived,
            limit=limit,
            start=start,
            expand=expand,
        )

        execution_time_ms = (time.time() - start_time) * 1000

        if return_enhanced_result:
            final_start = start or 0
            final_limit = limit or search_config.default_limit
            page_number = (final_start // final_limit) + 1 if final_limit > 0 else 1

            filters_applied = {
                "content_type": content_type,
                "space_key": space_key,
                "include_archived": include_archived,
                "hybrid_keyword_fetch_limit": search_config.hybrid_keyword_fetch_limit,
                "hybrid_semantic_fetch_limit": search_config.hybrid_semantic_fetch_limit,
                "hybrid_rrf_k": search_config.hybrid_rrf_k,
            }
            sort_criteria_applied = [{"field": "rrf_score", "direction": "desc"}]

            return self._process_search_result(
                final_search_result_obj,
                execution_time_ms=execution_time_ms,
                query=sanitized_text,
                filters=filters_applied,
                sort_criteria=sort_criteria_applied,
                current_page=page_number,
            )
        else:
            return cast(SearchResult_T, final_search_result_obj)
