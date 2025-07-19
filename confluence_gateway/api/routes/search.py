from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from confluence_gateway.api.dependencies import get_search_service
from confluence_gateway.api.schemas.requests import (
    AdvancedSearchRequest,
    CQLSearchRequest,
    SemanticSearchRequest,
)
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    PaginationLinks,
    SearchResponse,
    SearchResultItem,
    SemanticSearchResponse,
)
from confluence_gateway.core.exception_mapping import APIExceptionHandler
from confluence_gateway.core.transformers import ResponseTransformer
from confluence_gateway.services.search import SearchService

router = APIRouter()


def _build_search_response(
    search_result: Any, search_service: Any, request: Request | None = None
) -> SearchResponse:
    # Extract pagination parameters
    limit = getattr(search_result.results, "limit", 1) or 1
    start = getattr(search_result.results, "start", 0) or 0
    took_ms = getattr(search_result.statistics, "execution_time_ms", 0) or 0

    # Build query parameters for pagination links
    query_params = {}
    base_url = None
    if request:
        base_url = str(request.url).split("?")[0]
        for key, value in request.query_params.items():
            query_params[key] = value

    # Use shared transformer to build response data
    response_data = ResponseTransformer.build_search_response_data(
        search_result=search_result,
        search_service=search_service,
        took_ms=took_ms,
        start=start,
        limit=limit,
        base_url=base_url,
        query_params=query_params,
    )

    # Convert to SearchResultItem objects
    search_items = [
        SearchResultItem(**item_data) for item_data in response_data["results"]
    ]

    # Build pagination links
    links = None
    if base_url and response_data.get("links"):
        links = PaginationLinks(
            next=response_data["links"].get("next"),
            previous=response_data["links"].get("previous"),
        )

    return SearchResponse(
        results=search_items,
        total=response_data["total"],
        start=response_data["start"],
        limit=response_data["limit"],
        took_ms=response_data["took_ms"],
        page_count=response_data["page_count"],
        current_page=response_data["current_page"],
        has_more=response_data["has_more"],
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
@APIExceptionHandler.handle_exceptions
async def search_content(
    request: Request,
    query: str = Query(..., description="Text to search for", min_length=2),
    space_key: str | None = Query(None, description="Filter by space key"),
    content_type: str | None = Query(
        None, description="Filter by content type (page, blogpost, attachment, comment)"
    ),
    include_archived: bool = Query(False, description="Include archived content"),
    limit: int | None = Query(None, description="Maximum number of results to return"),
    start: int | None = Query(0, description="Starting position for pagination"),
    expand: list[str] | None = Query(
        None, description="Fields to expand in the response"
    ),
    use_hybrid: bool = Query(
        False,
        description="Enable hybrid search (keyword + semantic with RRF re-ranking)",
    ),
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    if use_hybrid:
        search_result = search_service.search_hybrid(
            text=query,
            content_type=content_type,
            space_key=space_key,
            include_archived=include_archived,
            limit=limit,
            start=start,
            expand=expand,
            return_enhanced_result=True,
        )
    else:
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
    "/semantic",
    response_model=SemanticSearchResponse,
    summary="Perform semantic search using vector embeddings",
    description="Search for content based on semantic similarity using vector embeddings",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid search parameters"},
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during semantic search",
        },
        503: {
            "model": ErrorResponse,
            "description": "Semantic search service unavailable",
        },
    },
)
@APIExceptionHandler.handle_exceptions
async def semantic_search(
    request: Request,
    search_request: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SemanticSearchResponse:
    results, took_ms = search_service.search_semantic(
        query=search_request.query,
        top_k=search_request.top_k,
        filters=search_request.filters,
    )

    return SemanticSearchResponse(
        results=results,
        took_ms=took_ms,
        query=search_request.query,
    )


@router.post(
    "/advanced",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        503: {"model": ErrorResponse, "description": "Confluence connection error"},
    },
)
@APIExceptionHandler.handle_exceptions
async def advanced_search(
    request: Request,
    search_request: AdvancedSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    if getattr(search_request, "use_hybrid", False):
        search_result = search_service.search_hybrid(
            text=search_request.query,
            content_type=search_request.content_type,
            space_key=search_request.space_key,
            include_archived=search_request.include_archived or False,
            limit=search_request.limit,
            start=search_request.start,
            expand=search_request.expand,
            return_enhanced_result=True,
        )
    else:
        search_result = search_service.search_by_text(
            text=search_request.query,
            content_type=search_request.content_type,
            space_key=search_request.space_key,
            include_archived=search_request.include_archived or False,
            limit=search_request.limit,
            start=search_request.start,
            expand=search_request.expand,
            get_all_results=search_request.get_all_results or False,
            max_results=search_request.max_results,
            min_relevance=search_request.min_relevance or 0.0,
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
@APIExceptionHandler.handle_exceptions
async def cql_search(
    request: Request,
    search_request: CQLSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    search_result = search_service.search_by_cql(
        cql=search_request.cql,
        limit=search_request.limit,
        start=search_request.start,
        expand=search_request.expand,
        return_enhanced_result=True,
    )

    return _build_search_response(search_result, search_service, request)
