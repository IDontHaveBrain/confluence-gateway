from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem


class PaginationLinks(BaseModel):
    next: Optional[str] = Field(None)
    previous: Optional[str] = Field(None)


class SearchResultItem(BaseModel):
    id: str = Field(...)
    title: str = Field(...)
    type: str = Field(...)
    space_key: str = Field(...)
    space_name: str = Field(...)
    url: str = Field(...)
    excerpt: Optional[str] = Field(None)
    last_modified: datetime = Field(...)

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
    results: list[SearchResultItem] = Field(default_factory=list)
    total: int = Field(...)
    start: int = Field(...)
    limit: int = Field(...)
    took_ms: float = Field(...)
    page_count: int = Field(...)
    current_page: int = Field(...)
    has_more: bool = Field(...)
    links: Optional[PaginationLinks] = Field(None)

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
    status: str = Field("error")
    code: int = Field(...)
    message: str = Field(...)
    details: Optional[dict[str, Any]] = Field(None)

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


class SemanticSearchResponse(BaseModel):
    results: list[VectorSearchResultItem] = Field(default_factory=list)
    took_ms: float = Field(...)
    query: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "id": "12345_chunk_0",
                            "score": 0.85,
                            "metadata": {
                                "original_content_id": "12345",
                                "title": "API Documentation",
                                "space_key": "DEV",
                                "url": "https://confluence.example.com/display/DEV/API+Documentation",
                                "chunk_sequence_number": 0,
                            },
                            "text": "This document describes the API endpoints...",
                        },
                        {
                            "id": "67890_chunk_2",
                            "score": 0.78,
                            "metadata": {
                                "original_content_id": "67890",
                                "title": "Getting Started with API",
                                "space_key": "DEV",
                                "url": "https://confluence.example.com/display/DEV/Getting+Started+with+API",
                                "chunk_sequence_number": 2,
                            },
                            "text": "...authentication is handled via API tokens...",
                        },
                    ],
                    "took_ms": 45.67,
                    "query": "how to use the confluence api",
                }
            ]
        }
    }
