from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem


class PaginationLinks(BaseModel):
    next: str | None = Field(None)
    previous: str | None = Field(None)


class SearchResultItem(BaseModel):
    id: str = Field(...)
    title: str = Field(...)
    type: str = Field(...)
    space_key: str = Field(...)
    space_name: str = Field(...)
    url: str = Field(...)
    excerpt: str | None = Field(None)
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


class IndexingStatusResponse(BaseModel):
    status: Literal["idle", "running", "success", "failure"] = Field(
        ..., description="Current status of the indexing process."
    )
    last_run_start_time: datetime | None = Field(
        None, description="Timestamp when the last indexing run started."
    )
    last_run_end_time: datetime | None = Field(
        None, description="Timestamp when the last indexing run finished."
    )
    last_error_message: str | None = Field(
        None, description="Error message from the last failed run, if any."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "idle",
                    "last_run_start_time": "2023-10-27T10:00:00Z",
                    "last_run_end_time": "2023-10-27T10:30:00Z",
                    "last_error_message": None,
                },
                {
                    "status": "running",
                    "last_run_start_time": "2023-10-27T11:00:00Z",
                    "last_run_end_time": None,
                    "last_error_message": None,
                },
                {
                    "status": "failure",
                    "last_run_start_time": "2023-10-27T09:00:00Z",
                    "last_run_end_time": "2023-10-27T09:15:00Z",
                    "last_error_message": "Failed to connect to Confluence API.",
                },
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
    links: PaginationLinks | None = Field(None)

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
    details: dict[str, Any] | None = Field(None)

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


class SourceDocument(BaseModel):
    id: str = Field(..., description="The unique ID of the source document chunk.")
    score: float = Field(
        ..., description="Relevance score of the chunk during retrieval."
    )
    title: str | None = Field(
        None, description="Title of the original Confluence page/attachment."
    )
    url: str | None = Field(
        None, description="URL of the original Confluence page/attachment."
    )
    space_key: str | None = Field(
        None, description="Space key of the original content."
    )

    @classmethod
    def from_vector_search_item(cls, item: VectorSearchResultItem) -> "SourceDocument":
        metadata = item.metadata or {}
        return cls(
            id=item.id,
            score=item.score,
            title=metadata.get("title"),
            url=metadata.get("url"),
            space_key=metadata.get("space_key"),
        )


class SpaceInfo(BaseModel):
    id: str = Field(..., description="The unique ID of the space.")
    key: str = Field(..., description="The unique key of the space.")
    name: str = Field(..., description="The display name of the space.")
    type: str = Field(..., description="The type of space (global or personal).")
    description: str | None = Field(None, description="The description of the space.")
    created_at: datetime | None = Field(None, description="When the space was created.")
    updated_at: datetime | None = Field(
        None, description="When the space was last updated."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "12345",
                    "key": "DEV",
                    "name": "Development",
                    "type": "global",
                    "description": "Space for development documentation and resources",
                    "created_at": "2023-01-15T10:00:00Z",
                    "updated_at": "2023-12-01T14:30:00Z",
                }
            ]
        }
    }


class GenerateAnswerResponse(BaseModel):
    answer: str = Field(
        ..., description="The generated answer based on the retrieved context."
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="List of source document chunks used to generate the answer.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Authentication is handled via API tokens passed in the Authorization header.",
                    "sources": [
                        {
                            "id": "12345_chunk_0",
                            "score": 0.85,
                            "title": "API Documentation",
                            "url": "https://confluence.example.com/display/DEV/API+Documentation",
                            "space_key": "DEV",
                        },
                        {
                            "id": "67890_chunk_2",
                            "score": 0.78,
                            "title": "Getting Started with API",
                            "url": "https://confluence.example.com/display/DEV/Getting+Started+with+API",
                            "space_key": "DEV",
                        },
                    ],
                }
            ]
        }
    }
