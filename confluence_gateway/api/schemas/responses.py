from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


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
