from typing import Optional

from pydantic import BaseModel, field_validator

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
