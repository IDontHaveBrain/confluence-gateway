import functools
import logging
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ContentType,
    SearchResult,
)
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.config import search_config
from confluence_gateway.core.exceptions import (
    SearchParameterError,
    SemanticSearchError,
)
from confluence_gateway.services.embedding import EmbeddingError, EmbeddingService
from confluence_gateway.services.indexing import IndexingService

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
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
    query: Optional[str] = None
    filters_applied: Optional[dict[str, Any]] = None
    sort_criteria: Optional[list[dict[str, str]]] = None

    def to_standard_result(self) -> SearchResult:
        return self.results


def validate_search_params(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        limit = kwargs.get("limit")
        original_start = kwargs.get("start")

        # Validate limit if explicitly provided (including zero)
        if limit is not None:
            if limit <= 0 or limit > search_config.max_limit:
                raise SearchParameterError(
                    f"Limit must be between 1 and {search_config.max_limit}"
                )
            actual_limit = limit
        else:
            # Apply default only if limit is None
            actual_limit = search_config.default_limit

        if original_start is not None and original_start < 0:
            raise SearchParameterError("Start position cannot be negative")
        actual_start = original_start if original_start is not None else 0

        kwargs["limit"] = actual_limit
        kwargs["start"] = actual_start

        return func(self, *args, **kwargs)

    return wrapper


class SearchService:
    def __init__(
        self,
        client: ConfluenceClient,
        indexing_service: Optional[IndexingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_db_adapter: Optional[VectorDBAdapter] = None,
    ):
        self.client = client
        self.indexing_service = indexing_service
        self.embedding_service = embedding_service
        self.vector_db_adapter = vector_db_adapter
        self.vector_index: Optional[VectorStoreIndex] = None

        if self.indexing_service:
            logger.info("SearchService initialized with IndexingService.")
        else:
            logger.warning("SearchService initialized WITHOUT IndexingService.")

        if self.embedding_service:
            logger.info("SearchService initialized with EmbeddingService.")
        else:
            logger.warning(
                "SearchService initialized WITHOUT EmbeddingService. Semantic search might be disabled."
            )

        if self.vector_db_adapter:
            logger.info("SearchService initialized with VectorDBAdapter.")
        else:
            logger.warning(
                "SearchService initialized WITHOUT VectorDBAdapter. Semantic search might be disabled."
            )

    def _prepare_sort_criteria(
        self,
        sort_by: Optional[list[Union[SortField, str]]],
        sort_direction: Optional[list[Union[SortDirection, str]]],
    ) -> Optional[list[dict[str, str]]]:
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

    def _sanitize_keywords(self, keywords: Union[str, list[str]]) -> str:
        if isinstance(keywords, str):
            return self._sanitize_text(keywords)

        if not keywords:
            raise SearchParameterError("Search keywords cannot be empty")

        sanitized_keywords = [self._sanitize_text(kw) for kw in keywords]
        return " ".join(sanitized_keywords)

    @validate_search_params
    def search_by_text(
        self,
        text: Union[str, list[str]],
        content_type: Optional[Union[ContentType, str]] = None,
        space_key: Optional[str] = None,
        include_archived: bool = False,
        limit: Optional[int] = None,
        start: Optional[int] = 0,
        expand: Optional[list[str]] = None,
        get_all_results: bool = False,
        max_results: Optional[int] = None,
        min_relevance: float = 0.0,
        top_n: Optional[int] = None,
        sort_by: Optional[list[Union[SortField, str]]] = None,
        sort_direction: Optional[list[Union[SortDirection, str]]] = None,
        return_enhanced_result: bool = True,
    ) -> SearchResult_T:
        sanitized_text = self._sanitize_keywords(text)

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
            return search_result

    def _sanitize_text(self, text: str) -> str:
        if not text:
            raise SearchParameterError("Search text cannot be empty")

        sanitized = re.sub(r"\s+", " ", text.strip())

        if not sanitized:
            raise SearchParameterError("Search text cannot be empty")

        if len(sanitized) < 2:
            raise SearchParameterError("Search text must be at least 2 characters long")

        sanitized = re.sub(
            r"[^\w\s\-.,;:!?\'\"()/+*=%&#@$^~]", "", sanitized, flags=re.UNICODE
        )

        return sanitized

    def _process_search_result(
        self,
        result: SearchResult,
        execution_time_ms: float = 0,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        sort_criteria: Optional[list[dict[str, str]]] = None,
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
        self, results: SearchResult, min_score: float = 0.0, top_n: Optional[int] = None
    ) -> SearchResult:
        filtered_results = list(results.results)

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
        sort_fields: list[Union[SortField, str]],
        directions: Optional[list[Union[SortDirection, str]]] = None,
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
        filters: Optional[dict[str, Any]] = None,
    ) -> tuple[list[VectorSearchResultItem], float]:
        if not self.embedding_service:
            logger.error(
                "Semantic search attempted but EmbeddingService is not available."
            )
            raise SemanticSearchError(
                "Semantic search is not configured: EmbeddingService is missing."
            )
        if not self.vector_db_adapter:
            logger.error(
                "Semantic search attempted but VectorDBAdapter is not available."
            )
            raise SemanticSearchError(
                "Semantic search is not configured: VectorDBAdapter is missing."
            )

        if not query or query.isspace():
            raise SearchParameterError("Semantic search query cannot be empty.")
        if top_k <= 0:
            raise SearchParameterError("top_k must be a positive integer.")

        sanitized_query = query.strip()
        logger.info(
            f"Performing semantic search for query: '{sanitized_query}', top_k={top_k}, filters={filters}"
        )

        start_time = time.time()

        try:
            logger.debug(f"Generating embedding for query: '{sanitized_query}'")
            query_embedding = self.embedding_service.embed_text(sanitized_query)
            if not query_embedding:
                # This might happen if the provider returns an empty list for some reason
                logger.error(
                    f"Embedding service returned an empty embedding for query: '{sanitized_query}'"
                )
                raise SemanticSearchError("Failed to generate a valid query embedding.")
            logger.debug("Query embedding generated successfully.")

        except EmbeddingError as e:
            logger.error(
                f"Embedding failed for query '{sanitized_query}': {e}", exc_info=True
            )
            raise SemanticSearchError(
                f"Failed to generate embedding for the query: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during query embedding: {e}", exc_info=True)
            raise SemanticSearchError(
                f"An unexpected error occurred during query embedding: {e}"
            ) from e

        try:
            logger.debug(
                f"Searching vector database with top_k={top_k} and filters={filters}"
            )
            results: list[VectorSearchResultItem] = self.vector_db_adapter.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            logger.debug(f"Vector database search returned {len(results)} results.")

        except Exception as e:
            # Catching generic Exception as adapter specifics might vary
            logger.error(f"Vector database search failed: {e}", exc_info=True)
            raise SemanticSearchError(
                f"Semantic search failed during vector database query: {e}"
            ) from e

        took_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Semantic search completed in {took_ms:.2f} ms, found {len(results)} results."
        )

        return results, took_ms

    @validate_search_params
    def search_by_cql(
        self,
        cql: str,
        limit: Optional[int] = None,
        start: Optional[int] = 0,
        expand: Optional[list[str]] = None,
        get_all_results: bool = False,
        max_results: Optional[int] = None,
        top_n: Optional[int] = None,
        sort_by: Optional[list[Union[SortField, str]]] = None,
        sort_direction: Optional[list[Union[SortDirection, str]]] = None,
        return_enhanced_result: bool = True,
    ) -> SearchResult_T:
        if not cql or not cql.strip():
            raise SearchParameterError("CQL query cannot be empty")

        equality_operators = ["=", "!=", "~", "^=", "$=", "*="]
        comparison_operators = ["<", ">", "<=", ">="]
        logical_operators = ["AND", "OR", "NOT"]

        field_operator_value_pattern = re.compile(
            r"\b[\w.-]+\s*("
            + "|".join(map(re.escape, equality_operators + comparison_operators))
            + r')\s*("([^"]|\\")*"|\'([^\']|\\\')*\'|\S+)',
            re.IGNORECASE,
        )

        logical_pattern = re.compile(
            r"\b("
            + "|".join(map(re.escape, logical_operators))
            + r")\b\s+"
            + r"("
            + r"\(|"
            + r"\b[\w.-]+\b\s*("
            + "|".join(map(re.escape, equality_operators + comparison_operators))
            + r")"
            + r")",
            re.IGNORECASE,
        )

        order_by_pattern = re.compile(
            r"\bORDER\s+BY\s+[\w.-]+(\s+(ASC|DESC))?", re.IGNORECASE
        )

        has_field_operator_value = bool(field_operator_value_pattern.search(cql))
        has_logical_structure = bool(logical_pattern.search(cql))
        has_order_by = bool(order_by_pattern.search(cql))
        has_parentheses = "(" in cql and ")" in cql

        if not (
            has_field_operator_value
            or has_logical_structure
            or has_order_by
            or has_parentheses
        ):
            raise SearchParameterError(
                "Invalid CQL query format. CQL must contain field-operator-value patterns, logical operators with conditions, ORDER BY clauses, or parentheses."
            )

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
            return search_result
