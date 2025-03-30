from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from confluence_gateway.core.config import search_config


class BaseSearchRequest(BaseModel):
    """Base model for search requests with common parameters."""

    limit: Optional[int] = None
    start: Optional[int] = None
    expand: Optional[list[str]] = None

    @field_validator("limit")
    def validate_limit(cls, v):
        if v is not None:
            if v <= 0:
                raise ValueError("Limit must be a positive integer")
            if v > search_config.max_limit:
                raise ValueError(f"Limit cannot exceed {search_config.max_limit}")
        return v

    @field_validator("start")
    def validate_start(cls, v):
        if v is not None and v < 0:
            raise ValueError("Start position cannot be negative")
        return v


class TextSearchRequest(BaseSearchRequest):
    """Model for text-based search requests."""

    query: str
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    include_archived: Optional[bool] = False

    @field_validator("query")
    def validate_query(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v):
        if v is not None:
            valid_types = ["page", "blogpost", "attachment", "comment"]
            if v.lower() not in valid_types:
                raise ValueError(
                    f"Invalid content type. Must be one of: {', '.join(valid_types)}"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "confluence api",
                    "space_key": "DEV",
                    "content_type": "page",
                    "include_archived": False,
                    "limit": 20,
                    "start": 0,
                }
            ]
        }
    }


class CQLSearchRequest(BaseSearchRequest):
    """Model for CQL-based search requests."""

    cql: str

    @field_validator("cql")
    def validate_cql(cls, v):
        if not v or not v.strip():
            raise ValueError("CQL query cannot be empty")

        # Check for basic CQL syntax indicators
        if not any(
            keyword in v.lower()
            for keyword in ["=", "~", "!=", ">=", "<=", "and", "or", "not"]
        ):
            raise ValueError("Invalid CQL query format")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"cql": "space = DEV AND type = page", "limit": 20, "start": 0}
            ]
        }
    }


class AdvancedSearchRequest(BaseSearchRequest):
    """Model for advanced text-based search requests."""

    query: str
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    include_archived: Optional[bool] = False
    get_all_results: Optional[bool] = False
    max_results: Optional[int] = None
    min_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_n: Optional[int] = Field(None, gt=0)
    sort_by: Optional[list[str]] = None
    sort_direction: Optional[list[str]] = None

    @field_validator("query")
    def validate_query(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v):
        if v is not None:
            valid_types = ["page", "blogpost", "attachment", "comment"]
            if v.lower() not in valid_types:
                raise ValueError(
                    f"Invalid content type. Must be one of: {', '.join(valid_types)}"
                )
        return v

    @field_validator("sort_by")
    def validate_sort_by(cls, v):
        if v is not None:
            valid_fields = ["title", "created_at", "updated_at", "score", "space_key"]
            for field in v:
                if field.lower() not in valid_fields:
                    raise ValueError(
                        f"Invalid sort field. Must be one of: {', '.join(valid_fields)}"
                    )
        return v

    @field_validator("sort_direction")
    def validate_sort_direction(cls, v):
        if v is not None:
            valid_directions = ["asc", "desc"]
            for direction in v:
                if direction.lower() not in valid_directions:
                    raise ValueError(
                        f"Invalid sort direction. Must be one of: {', '.join(valid_directions)}"
                    )
        return v

    @field_validator("max_results")
    def validate_max_results(cls, v, values):
        if v is not None:
            if v <= 0:
                raise ValueError("max_results must be a positive integer")
            if not values.data.get("get_all_results", False):
                raise ValueError(
                    "max_results can only be used when get_all_results is True"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "confluence api",
                    "space_key": "DEV",
                    "content_type": "page",
                    "include_archived": False,
                    "limit": 20,
                    "start": 0,
                    "get_all_results": False,
                    "min_relevance": 0.5,
                    "top_n": 10,
                    "sort_by": ["updated_at", "title"],
                    "sort_direction": ["desc", "asc"],
                }
            ]
        }
    }


class PaginationLinks(BaseModel):
    """Links for paginated results."""

    next: Optional[str] = Field(None, description="URL for the next page of results")
    previous: Optional[str] = Field(
        None, description="URL for the previous page of results"
    )


class SearchResultItem(BaseModel):
    """Individual item in search results."""

    id: str = Field(..., description="Unique identifier of the content")
    title: str = Field(..., description="Title of the content")
    type: str = Field(
        ..., description="Type of content (page, blogpost, attachment, comment)"
    )
    space_key: str = Field(..., description="Key of the space containing the content")
    space_name: str = Field(..., description="Name of the space containing the content")
    url: str = Field(..., description="URL to view the content")
    excerpt: Optional[str] = Field(
        None, description="Text excerpt with highlighted matches"
    )
    last_modified: datetime = Field(..., description="Last modification timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345",
                    "title": "API Documentation",
                    "type": "page",
                    "space_key": "DEV",
                    "space_name": "Development",
                    "url": "https://confluence.example.com/display/DEV/API+Documentation",
                    "excerpt": "This document describes the <em>API</em> endpoints...",
                    "last_modified": "2023-05-15T14:32:21Z",
                }
            ]
        }
    }


class SearchResponse(BaseModel):
    """Response model for search requests."""

    results: list[SearchResultItem] = Field(
        default_factory=list, description="List of search result items"
    )
    total: int = Field(..., description="Total number of results available")
    start: int = Field(..., description="Starting index of results")
    limit: int = Field(..., description="Maximum number of results returned")
    took_ms: float = Field(
        ..., description="Time taken to execute the search in milliseconds"
    )
    page_count: int = Field(..., description="Total number of pages available")
    current_page: int = Field(..., description="Current page number")
    has_more: bool = Field(..., description="Whether there are more results available")
    links: Optional[PaginationLinks] = Field(None, description="Pagination links")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "id": "12345",
                            "title": "API Documentation",
                            "type": "page",
                            "space_key": "DEV",
                            "space_name": "Development",
                            "url": "https://confluence.example.com/display/DEV/API+Documentation",
                            "excerpt": "This document describes the <em>API</em> endpoints...",
                            "last_modified": "2023-05-15T14:32:21Z",
                        }
                    ],
                    "total": 42,
                    "start": 0,
                    "limit": 20,
                    "took_ms": 123.45,
                    "page_count": 3,
                    "current_page": 1,
                    "has_more": True,
                    "links": {
                        "next": "/api/search?query=api&start=20&limit=20",
                        "previous": None,
                    },
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    status: str = Field("error", description="Status of the response")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(
        None, description="Additional error details"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "error",
                    "code": 400,
                    "message": "Invalid search parameters",
                    "details": {
                        "param": "query",
                        "reason": "Query must be at least 2 characters long",
                    },
                }
            ]
        }
    }
