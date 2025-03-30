import functools
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.config import search_config
from confluence_gateway.core.exceptions import SearchParameterError

T = TypeVar("T")
R = TypeVar("R")
SearchResult_T = Union[SearchResult, "EnhancedSearchResult"]


class SortDirection(str, Enum):
    """Sort direction for search results."""

    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    """Fields to sort search results by."""

    TITLE = "title"
    CREATED = "created_at"
    UPDATED = "updated_at"
    RELEVANCE = "score"
    SPACE = "space_key"


class SearchStatistics(BaseModel):
    """Statistics about a search operation."""

    total_results: int = 0
    filtered_results: int = 0
    total_pages: int = 0
    current_page: int = 0
    execution_time_ms: float = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class EnhancedSearchResult(BaseModel):
    """Enhanced search result with additional metadata."""

    results: SearchResult
    statistics: SearchStatistics
    query: Optional[str] = None
    filters_applied: Optional[dict[str, Any]] = None
    sort_criteria: Optional[list[dict[str, str]]] = None

    def to_standard_result(self) -> SearchResult:
        """Convert back to standard SearchResult."""
        return self.results


def validate_search_params(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to validate common search parameters.

    This ensures consistent parameter validation across all search methods.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with parameter validation
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract parameters that need validation
        limit = kwargs.get("limit")
        start = kwargs.get("start", 0)

        # Apply default parameters
        actual_limit = limit or search_config.default_limit
        actual_start = start or 0

        # Validate parameter ranges
        if actual_limit <= 0 or actual_limit > search_config.max_limit:
            raise SearchParameterError(
                f"Limit must be between 1 and {search_config.max_limit}"
            )

        if actual_start < 0:
            raise SearchParameterError("Start position cannot be negative")

        # Update kwargs with validated parameters
        kwargs["limit"] = actual_limit
        kwargs["start"] = actual_start

        return func(self, *args, **kwargs)

    return wrapper


class SearchService:
    """Service for searching Confluence content."""

    def __init__(self, client: ConfluenceClient):
        """
        Initialize the search service.

        Args:
            client: ConfluenceClient instance for API communication
        """
        self.client = client

    def _prepare_sort_criteria(
        self,
        sort_by: Optional[list[Union[SortField, str]]],
        sort_direction: Optional[list[Union[SortDirection, str]]],
    ) -> Optional[list[dict[str, str]]]:
        """
        Prepare sort criteria metadata from sort parameters.

        Args:
            sort_by: Fields to sort results by
            sort_direction: Sort directions corresponding to sort_by fields

        Returns:
            List of dictionaries with field and direction, or None if sort_by is None
        """
        if not sort_by:
            return None

        sort_criteria = []
        directions = sort_direction or []

        for i, field in enumerate(sort_by):
            direction = directions[i] if i < len(directions) else SortDirection.ASC

            # Normalize field to enum
            if isinstance(field, str):
                try:
                    field = SortField(field.lower())
                except ValueError:
                    field = SortField.TITLE  # Default to title if invalid

            # Normalize direction to enum
            if isinstance(direction, str):
                try:
                    direction = SortDirection(direction.lower())
                except ValueError:
                    direction = SortDirection.ASC  # Default to ascending if invalid

            sort_criteria.append(
                {"field": str(field.value), "direction": str(direction.value)}
            )

        return sort_criteria

    @validate_search_params
    def search_by_text(
        self,
        text: str,
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
        """
        Search Confluence content using text query.

        Args:
            text: Text to search for
            content_type: Filter by content type (page, blogpost, etc.)
            space_key: Filter by space key
            include_archived: Include archived content (default: False)
            limit: Maximum number of results per page (default: from config)
            start: Starting position (default: 0)
            expand: Fields to expand in the response (default: from config)
            get_all_results: Whether to fetch all pages of results (default: False)
            max_results: Maximum total results to fetch when get_all_results is True
            min_relevance: Minimum relevance score (0.0-1.0) for filtering results
            top_n: Number of top results to return
            sort_by: Fields to sort results by
            sort_direction: Sort directions corresponding to sort_by fields
            return_enhanced_result: Whether to return enhanced result with metadata

        Returns:
            SearchResult or EnhancedSearchResult object containing the search results

        Raises:
            SearchParameterError: If the search text is invalid or empty
        """
        # 1. Sanitize input text
        sanitized_text = self._sanitize_text(text)

        # 2. Handle expand parameter
        actual_expand = expand
        if actual_expand is None and search_config.default_expand:
            actual_expand = search_config.default_expand

        # 3. Create filters dictionary for metadata
        filters = {
            "content_type": content_type,
            "space_key": space_key,
            "include_archived": include_archived,
            "min_relevance": min_relevance,
            "top_n": top_n,
        }

        # 4. Prepare sort criteria for metadata
        sort_criteria = self._prepare_sort_criteria(sort_by, sort_direction)

        # 5. Calculate pagination
        page_number = ((start or 0) // (limit or 1)) + 1 if (limit or 0) > 0 else 1

        # 6. Measure execution time
        start_time = time.time()

        # 7. Call client's search method with sanitized parameters
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

        # 8. Apply relevance filtering if specified
        if min_relevance > 0 or top_n is not None:
            search_result = self._filter_by_relevance(
                search_result, min_score=min_relevance, top_n=top_n
            )

        # 9. Apply sorting if specified
        if sort_by:
            search_result = self._sort_results(
                search_result, sort_fields=sort_by, directions=sort_direction
            )

        # 10. Process and enhance result
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
        """
        Sanitize the search text by removing excessive whitespace and special characters.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text

        Raises:
            SearchParameterError: If the text is empty or invalid
        """
        if not text:
            raise SearchParameterError("Search text cannot be empty")

        # Trim whitespace and normalize spaces
        sanitized = re.sub(r"\s+", " ", text.strip())

        if not sanitized:
            raise SearchParameterError("Search text cannot be empty")

        if len(sanitized) < 2:
            raise SearchParameterError("Search text must be at least 2 characters long")

        # Keep alphanumeric (including Unicode), spaces, punctuation, and common technical symbols
        # Remove characters that might interfere with Confluence search syntax
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
        """
        Process and enhance the search result with additional metadata.

        Args:
            result: The original search result from the client
            execution_time_ms: Time taken to execute the search in milliseconds
            query: Original search query
            filters: Filters applied to the search
            sort_criteria: Sorting criteria applied
            current_page: Current page number

        Returns:
            Enhanced search result with metadata
        """
        # Calculate pagination information
        items_per_page = result.limit or search_config.default_limit
        total_pages = (
            (result.total_size + items_per_page - 1) // items_per_page
            if items_per_page > 0
            else 0
        )

        # Create statistics
        statistics = SearchStatistics(
            total_results=result.total_size,
            filtered_results=len(result.results),
            total_pages=total_pages,
            current_page=current_page,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
        )

        # Create enhanced result
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
        """
        Filter search results based on relevance score.

        Currently, this method only implements a simple top-N filter as Confluence API
        doesn't provide explicit relevance scores. In the future, this could be extended to:

        1. Calculate custom relevance scores based on keyword frequency
        2. Use ML-based approaches to rank results by relevance to query
        3. Re-rank based on content freshness, popularity, or user preferences
        4. Apply domain-specific boosting factors

        Args:
            results: Original search results
            min_score: Minimum relevance score to include (0.0-1.0)
                       Currently not implemented in basic version
            top_n: Number of top results to keep

        Returns:
            Filtered search results
        """
        filtered_results = list(results.results)

        # TODO: Future enhancement - Implement custom relevance scoring
        # This would involve text analysis of the content against the query
        # For now, we assume Confluence's ordering is by relevance

        # If min_score is specified, we could filter here
        # Currently a placeholder since we don't have actual scores

        # If top_n is specified, limit to the top N results
        if top_n is not None and top_n > 0 and len(filtered_results) > top_n:
            filtered_results = filtered_results[:top_n]

        # Create a new SearchResult with the filtered items
        return SearchResult(
            total_size=results.total_size,  # Keep original total for reference
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
        """
        Sort search results by specified criteria.

        Args:
            results: Search results to sort
            sort_fields: Fields to sort by (in priority order)
            directions: Sort directions (asc/desc) corresponding to sort_fields

        Returns:
            Sorted search results
        """
        if not results.results or not sort_fields:
            return results

        # Prepare directions list, defaulting to ASC if not specified
        if directions is None:
            directions = [SortDirection.ASC] * len(sort_fields)
        elif len(directions) < len(sort_fields):
            # Pad with ASC if not enough directions
            directions.extend(
                [SortDirection.ASC] * (len(sort_fields) - len(directions))
            )

        # Convert string values to enums if needed
        normalized_fields = []
        for field in sort_fields:
            if isinstance(field, str):
                try:
                    field = SortField(field.lower())
                except ValueError:
                    field = SortField.TITLE  # Default to title if invalid
            normalized_fields.append(field)

        normalized_directions = []
        for direction in directions:
            if isinstance(direction, str):
                try:
                    direction = SortDirection(direction.lower())
                except ValueError:
                    direction = SortDirection.ASC  # Default to ascending if invalid
            normalized_directions.append(direction)

        # Create a list of items to sort
        items_to_sort = list(results.results)

        # Sort the items using a multi-tier approach for better control over sort order
        for sort_idx in range(len(normalized_fields) - 1, -1, -1):
            field = normalized_fields[sort_idx]
            direction = normalized_directions[sort_idx]
            reverse = direction == SortDirection.DESC

            def get_sort_value(item: ConfluencePage) -> Any:
                """Extract the sort value from the item based on field."""
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
                else:  # Default or RELEVANCE - keep original order
                    return 0

            # Custom key function that properly handles different data types
            def key_func(item: ConfluencePage) -> Any:
                value = get_sort_value(item)
                # No need to transform the value for sorting
                # Python's sorted handles reverse parameter separately
                return value

            # Sort in place with proper type handling
            items_to_sort.sort(key=key_func, reverse=reverse)

        # Create a new SearchResult with sorted items
        return SearchResult(
            total_size=results.total_size,
            start=results.start,
            limit=results.limit,
            results=items_to_sort,
        )

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
        """
        Search Confluence content using CQL (Confluence Query Language).

        Args:
            cql: CQL query string
            limit: Maximum number of results per page (default: from config)
            start: Starting position (default: 0)
            expand: Fields to expand in the response (default: from config)
            get_all_results: Whether to fetch all pages of results (default: False)
            max_results: Maximum total results to fetch when get_all_results is True
            top_n: Number of top results to return
            sort_by: Fields to sort results by
            sort_direction: Sort directions corresponding to sort_by fields
            return_enhanced_result: Whether to return enhanced result with metadata

        Returns:
            SearchResult or EnhancedSearchResult object containing the search results

        Raises:
            SearchParameterError: If the CQL query is invalid or empty
        """
        # 1. Validate CQL query
        if not cql or not cql.strip():
            raise SearchParameterError("CQL query cannot be empty")

        # Check for basic CQL syntax indicators
        if not any(
            keyword in cql.lower()
            for keyword in ["=", "~", "!=", ">=", "<=", "and", "or", "not"]
        ):
            raise SearchParameterError("Invalid CQL query format")

        # 2. Handle expand parameter
        actual_expand = expand
        if actual_expand is None and search_config.default_expand:
            actual_expand = search_config.default_expand

        # 3. Create filters dictionary for metadata
        filters = {"cql": cql, "top_n": top_n}

        # 4. Prepare sort criteria for metadata
        sort_criteria = self._prepare_sort_criteria(sort_by, sort_direction)

        # 5. Calculate pagination
        page_number = ((start or 0) // (limit or 1)) + 1 if (limit or 0) > 0 else 1

        # 6. Measure execution time
        start_time = time.time()

        # 7. Use the client's public method for CQL search
        search_result = self.client.search_by_cql(
            cql=cql,
            limit=limit,
            start=start,
            expand=actual_expand,
            get_all_results=get_all_results,
            max_results=max_results,
        )

        execution_time_ms = (time.time() - start_time) * 1000

        # 8. Apply top-N filtering if specified
        if top_n is not None:
            search_result = self._filter_by_relevance(search_result, top_n=top_n)

        # 9. Apply sorting if specified
        if sort_by:
            search_result = self._sort_results(
                search_result, sort_fields=sort_by, directions=sort_direction
            )

        # 10. Process and enhance result
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
