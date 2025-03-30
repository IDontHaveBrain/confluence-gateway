from functools import wraps
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from confluence_gateway.api.app import get_search_service
from confluence_gateway.api.schemas.requests import (
    AdvancedSearchRequest,
    CQLSearchRequest,
)
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    PaginationLinks,
    SearchResponse,
    SearchResultItem,
)
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    SearchParameterError,
)
from confluence_gateway.services.search import SearchService

router = APIRouter()


def handle_search_exceptions(func):
    """Decorator to handle common exceptions in search endpoints."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SearchParameterError as e:
            error = ErrorResponse(
                code=400, message=str(e), details={"type": "search_parameter_error"}
            )
            raise HTTPException(status_code=400, detail=error.model_dump())
        except ConfluenceAuthenticationError as e:
            error = ErrorResponse(
                code=401, message=str(e), details={"type": "authentication_error"}
            )
            raise HTTPException(status_code=401, detail=error.model_dump())
        except ConfluenceConnectionError as e:
            error = ErrorResponse(
                code=503,
                message=f"Confluence connection error: {str(e)}",
                details={
                    "type": "connection_error",
                    "cause": str(getattr(e, "cause", "")),
                },
            )
            raise HTTPException(status_code=503, detail=error.model_dump())
        except ConfluenceAPIError as e:
            status_code = e.status_code or 500
            error = ErrorResponse(
                code=status_code,
                message=e.error_message or str(e),
                details={"type": "api_error", "status_code": status_code},
            )
            raise HTTPException(status_code=status_code, detail=error.model_dump())
        except Exception as e:
            error = ErrorResponse(
                code=500,
                message=f"Unexpected error: {str(e)}",
                details={"type": "server_error"},
            )
            raise HTTPException(status_code=500, detail=error.model_dump())

    return wrapper


def _build_search_response(
    search_result, search_service, request: Optional[Request] = None
) -> SearchResponse:
    """Convert a search_result to SearchResponse with pagination metadata."""
    # More efficiently build search items with list comprehension
    search_items = [
        SearchResultItem(
            id=item.id,
            title=item.title,
            type=search_service.client.extract_content_fields(item).get(
                "type", str(item.content_type)
            ),
            space_key=search_service.client.extract_content_fields(item).get(
                "space_key", ""
            ),
            space_name=search_service.client.extract_content_fields(item).get(
                "space_name", ""
            ),
            url=search_service.client.extract_content_fields(item).get("url")
            or f"{search_service.client.base_url}/wiki/spaces/{search_service.client.extract_content_fields(item).get('space_key', '')}/pages/{item.id}",
            excerpt=getattr(item, "excerpt", None),
            last_modified=item.updated_at or item.created_at,
        )
        for item in search_result.results.results
    ]

    # Calculate pagination metadata
    limit = search_result.results.limit or 1  # Avoid division by zero
    start = search_result.results.start or 0
    total = search_result.statistics.total_results

    current_page = (start // limit) + 1
    page_count = (total + limit - 1) // limit if limit > 0 else 0
    has_more = current_page < page_count

    # Build pagination links if we have request data
    links = None
    if request:
        base_url = str(request.url).split("?")[0]

        # Get current query params
        params = {}
        for key, value in request.query_params.items():
            params[key] = value

        # Build links
        links = PaginationLinks()

        # Next page link
        if has_more:
            next_params = params.copy()
            next_params["start"] = str(start + limit)
            links.next = f"{base_url}?{urlencode(next_params)}"

        # Previous page link
        if start > 0:
            prev_params = params.copy()
            prev_params["start"] = str(max(0, start - limit))
            links.previous = f"{base_url}?{urlencode(prev_params)}"

    return SearchResponse(
        results=search_items,
        total=total,
        start=start,
        limit=limit,
        took_ms=search_result.statistics.execution_time_ms,
        page_count=page_count,
        current_page=current_page,
        has_more=has_more,
        links=links,
    )


@router.get(
    "",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Confluence connection error"},
    },
)
@handle_search_exceptions
async def search_content(
    request: Request,
    query: str = Query(..., description="Text to search for", min_length=2),
    space_key: Optional[str] = Query(None, description="Filter by space key"),
    content_type: Optional[str] = Query(
        None, description="Filter by content type (page, blogpost, attachment, comment)"
    ),
    include_archived: bool = Query(False, description="Include archived content"),
    limit: Optional[int] = Query(
        None, description="Maximum number of results to return"
    ),
    start: Optional[int] = Query(0, description="Starting position for pagination"),
    expand: Optional[list[str]] = Query(
        None, description="Fields to expand in the response"
    ),
    search_service: SearchService = Depends(get_search_service),
):
    """
    Search Confluence content using text query.

    Returns paginated search results with links to next/previous pages when available.
    """
    search_result = search_service.search_by_text(
        text=query,
        content_type=content_type,
        space_key=space_key,
        include_archived=include_archived,
        limit=limit,
        start=start,
        expand=expand,
        return_enhanced_result=True,
    )

    return _build_search_response(search_result, search_service, request)


@router.post(
    "/advanced",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Confluence connection error"},
    },
)
@handle_search_exceptions
async def advanced_search(
    request: Request,
    search_request: AdvancedSearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    """
    Advanced search for Confluence content with additional filtering and sorting options.

    Allows more complex search queries with relevance filtering and custom sorting.
    """
    search_result = search_service.search_by_text(
        text=search_request.query,
        content_type=search_request.content_type,
        space_key=search_request.space_key,
        include_archived=search_request.include_archived,
        limit=search_request.limit,
        start=search_request.start,
        expand=search_request.expand,
        get_all_results=search_request.get_all_results,
        max_results=search_request.max_results,
        min_relevance=search_request.min_relevance,
        top_n=search_request.top_n,
        sort_by=search_request.sort_by,
        sort_direction=search_request.sort_direction,
        return_enhanced_result=True,
    )

    return _build_search_response(search_result, search_service, request)


@router.post(
    "/cql",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid CQL query"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Confluence connection error"},
    },
)
@handle_search_exceptions
async def cql_search(
    request: Request,
    search_request: CQLSearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    """
    Search Confluence content using CQL (Confluence Query Language).

    Provides direct access to Confluence's powerful CQL search capabilities
    for advanced users who need precise control over their search queries.
    """
    search_result = search_service.search_by_cql(
        cql=search_request.cql,
        limit=search_request.limit,
        start=search_request.start,
        expand=search_request.expand,
        return_enhanced_result=True,
    )

    return _build_search_response(search_result, search_service, request)
