from typing import Any

from pydantic import BaseModel, Field, field_validator

from confluence_gateway.core.config import search_config


class BaseSearchRequest(BaseModel):
    limit: int | None = None
    start: int | None = None
    expand: list[str] | None = None

    @field_validator("limit")
    def validate_limit(cls, v: int | None) -> int | None:
        if v is not None:
            if v <= 0:
                raise ValueError("Limit must be a positive integer")
            if v > search_config.max_limit:
                raise ValueError(f"Limit cannot exceed {search_config.max_limit}")
        return v

    @field_validator("start")
    def validate_start(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("Start position cannot be negative")
        return v


class TextSearchRequest(BaseSearchRequest):
    query: str
    space_key: str | None = None
    content_type: str | None = None
    include_archived: bool | None = False

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v: str | None) -> str | None:
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


class IndexingTriggerRequest(BaseModel):
    space_keys: list[str] | None = Field(
        None,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"space_keys": ["DEV", "PROD"]},
                {},
            ]
        }
    }


class CQLSearchRequest(BaseSearchRequest):
    cql: str

    @field_validator("cql")
    def validate_cql(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("CQL query cannot be empty")

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
    query: str
    space_key: str | None = None
    content_type: str | None = None
    include_archived: bool | None = False
    get_all_results: bool | None = False
    max_results: int | None = None
    min_relevance: float | None = Field(None, ge=0.0, le=1.0)
    top_n: int | None = Field(None, gt=0)
    sort_by: list[str] | None = None
    sort_direction: list[str] | None = None
    use_hybrid: bool | None = Field(
        False,
        description="Enable hybrid search (keyword + semantic with RRF re-ranking)",
    )

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v

    @field_validator("content_type")
    def validate_content_type(cls, v: str | None) -> str | None:
        if v is not None:
            valid_types = ["page", "blogpost", "attachment", "comment"]
            if v.lower() not in valid_types:
                raise ValueError(
                    f"Invalid content type. Must be one of: {', '.join(valid_types)}"
                )
        return v

    @field_validator("sort_by")
    def validate_sort_by(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            valid_fields = ["title", "created_at", "updated_at", "score", "space_key"]
            for field in v:
                if field.lower() not in valid_fields:
                    raise ValueError(
                        f"Invalid sort field. Must be one of: {', '.join(valid_fields)}"
                    )
        return v

    @field_validator("sort_direction")
    def validate_sort_direction(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            valid_directions = ["asc", "desc"]
            for direction in v:
                if direction.lower() not in valid_directions:
                    raise ValueError(
                        f"Invalid sort direction. Must be one of: {', '.join(valid_directions)}"
                    )
        return v

    @field_validator("max_results")
    def validate_max_results(cls, v: int | None, values: Any) -> int | None:
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
                    "use_hybrid": False,
                }
            ]
        }
    }


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, gt=0)
    filters: dict[str, Any] | None = None

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "how to use the confluence api",
                    "top_k": 5,
                    "filters": {"space_key": "DEV"},
                }
            ]
        }
    }


class GenerateAnswerRequest(BaseModel):
    query: str = Field(...)
    top_k_retrieval: int | None = Field(
        default=5,
        gt=0,
    )
    filters: dict[str, Any] | None = Field(
        default=None,
    )

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Query must be at least 2 characters long")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How is authentication handled in the API?",
                    "top_k_retrieval": 5,
                    "filters": {"space_key": "DEV"},
                }
            ]
        }
    }
