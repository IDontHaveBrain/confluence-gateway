"""Shared validation rules and utilities for API and CLI interfaces."""

from __future__ import annotations

import json
from typing import Any

from confluence_gateway.core.exceptions import SearchParameterError


class ValidationUtils:
    """Shared validation utilities used by both API and CLI interfaces."""

    # Valid enum values for various parameters
    VALID_CONTENT_TYPES = ["page", "blogpost", "attachment", "comment"]
    VALID_SORT_FIELDS = ["title", "created_at", "updated_at", "score", "space_key"]
    VALID_SORT_DIRECTIONS = ["asc", "desc"]
    VALID_SPACE_TYPES = ["personal", "global", "all"]
    VALID_SPACE_SORT_FIELDS = ["name", "key", "type", "id"]

    # Validation constraints
    MIN_QUERY_LENGTH = 2
    MIN_MAX_RESULTS = 1
    MAX_MAX_RESULTS = 1000

    @classmethod
    def validate_query(cls, query: str, field_name: str = "query") -> str:
        """Validate search query length and format."""
        if not query or len(query.strip()) < cls.MIN_QUERY_LENGTH:
            raise SearchParameterError(
                f"{field_name.title()} must be at least {cls.MIN_QUERY_LENGTH} characters long"
            )
        return query.strip()

    @classmethod
    def validate_json_string(
        cls, json_string: str, field_name: str = "json"
    ) -> dict[str, Any]:
        """Validate and parse JSON string into dictionary."""
        try:
            parsed = json.loads(json_string)
            if not isinstance(parsed, dict):
                raise SearchParameterError(
                    f"{field_name.title()} must be a valid JSON object string."
                )
            return parsed
        except json.JSONDecodeError as e:
            raise SearchParameterError(f"Invalid JSON in {field_name} string: {e}")

    @classmethod
    def validate_content_type(
        cls, content_type: str | None, field_name: str = "content_type"
    ) -> str | None:
        """Validate content type against allowed values."""
        if content_type is not None:
            if content_type not in cls.VALID_CONTENT_TYPES:
                raise SearchParameterError(
                    f"Invalid {field_name}: '{content_type}'. "
                    f"Valid options: {', '.join(cls.VALID_CONTENT_TYPES)}"
                )
        return content_type

    @classmethod
    def validate_sort_fields(
        cls, sort_fields: list[str] | None, field_name: str = "sort_by"
    ) -> list[str] | None:
        """Validate sort fields against allowed values."""
        if sort_fields is not None:
            invalid_fields = [
                field for field in sort_fields if field not in cls.VALID_SORT_FIELDS
            ]
            if invalid_fields:
                raise SearchParameterError(
                    f"Invalid {field_name}: {', '.join(invalid_fields)}. "
                    f"Valid options: {', '.join(cls.VALID_SORT_FIELDS)}"
                )
        return sort_fields

    @classmethod
    def validate_sort_directions(
        cls, sort_directions: list[str] | None, field_name: str = "sort_direction"
    ) -> list[str] | None:
        """Validate sort directions against allowed values."""
        if sort_directions is not None:
            invalid_directions = [
                direction
                for direction in sort_directions
                if direction not in cls.VALID_SORT_DIRECTIONS
            ]
            if invalid_directions:
                raise SearchParameterError(
                    f"Invalid {field_name}: {', '.join(invalid_directions)}. "
                    f"Valid options: {', '.join(cls.VALID_SORT_DIRECTIONS)}"
                )
        return sort_directions

    @classmethod
    def validate_max_results(
        cls, max_results: int | None, field_name: str = "max_results"
    ) -> int | None:
        """Validate max_results parameter."""
        if max_results is not None:
            if max_results <= 0:
                raise SearchParameterError(f"{field_name} must be greater than 0")
            if max_results > cls.MAX_MAX_RESULTS:
                raise SearchParameterError(
                    f"{field_name} cannot exceed {cls.MAX_MAX_RESULTS}"
                )
        return max_results

    @classmethod
    def validate_cql_query(cls, cql: str, field_name: str = "cql") -> str:
        """Validate CQL query format."""
        if not cql or not cql.strip():
            raise SearchParameterError(f"{field_name} query cannot be empty")
        return cql.strip()

    @classmethod
    def validate_mutually_exclusive(
        cls, value1: Any, value2: Any, field1_name: str, field2_name: str
    ) -> None:
        """Validate that two fields are mutually exclusive."""
        if value1 and value2:
            raise SearchParameterError(
                f"Cannot use {field1_name} and {field2_name} together. Please choose one."
            )

    @classmethod
    def validate_space_type(
        cls, space_type: str | None, field_name: str = "type"
    ) -> str | None:
        """Validate space type parameter for spaces commands."""
        if space_type is not None:
            space_type_lower = space_type.lower()
            if space_type_lower not in cls.VALID_SPACE_TYPES:
                raise SearchParameterError(
                    f"Invalid {field_name}: '{space_type}'. "
                    f"Valid options: {', '.join(cls.VALID_SPACE_TYPES)}"
                )
            return space_type_lower
        return space_type

    @classmethod
    def validate_space_sort_field(
        cls, sort_field: str | None, field_name: str = "sort"
    ) -> str | None:
        """Validate space sort field parameter for spaces commands."""
        if sort_field is not None:
            sort_field_lower = sort_field.lower()
            if sort_field_lower not in cls.VALID_SPACE_SORT_FIELDS:
                raise SearchParameterError(
                    f"Invalid {field_name}: '{sort_field}'. "
                    f"Valid options: {', '.join(cls.VALID_SPACE_SORT_FIELDS)}"
                )
            return sort_field_lower
        return sort_field


class ParameterValidator:
    """Higher-level parameter validation for common use cases."""

    @staticmethod
    def validate_search_request_params(
        query: str,
        content_type: str | None = None,
        sort_by: list[str] | None = None,
        sort_direction: list[str] | None = None,
        max_results: int | None = None,
        filters: str | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize search request parameters."""
        validated_params: dict[str, Any] = {}

        # Validate query
        validated_params["query"] = ValidationUtils.validate_query(query)

        # Validate optional parameters
        validated_params["content_type"] = ValidationUtils.validate_content_type(
            content_type
        )
        validated_params["sort_by"] = ValidationUtils.validate_sort_fields(sort_by)
        validated_params["sort_direction"] = ValidationUtils.validate_sort_directions(
            sort_direction
        )
        validated_params["max_results"] = ValidationUtils.validate_max_results(
            max_results
        )

        # Validate and parse filters JSON if provided
        if filters:
            validated_params["filters"] = ValidationUtils.validate_json_string(
                filters, "filters"
            )
        else:
            validated_params["filters"] = None

        return validated_params

    @staticmethod
    def validate_indexing_request_params(
        space_keys: list[str] | None = None,
        index_all: bool = False,
    ) -> dict[str, Any]:
        """Validate and normalize indexing request parameters."""
        # Check mutually exclusive options
        ValidationUtils.validate_mutually_exclusive(
            index_all, space_keys, "index_all", "space_keys"
        )

        return {
            "space_keys": space_keys,
            "index_all": index_all,
        }

    @staticmethod
    def validate_generation_request_params(
        query: str,
        max_results: int | None = None,
        filters: str | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize generation request parameters."""
        validated_params: dict[str, Any] = {}

        # Validate query
        validated_params["query"] = ValidationUtils.validate_query(query)

        # Validate optional parameters
        validated_params["max_results"] = ValidationUtils.validate_max_results(
            max_results
        )

        # Validate and parse filters JSON if provided
        if filters:
            validated_params["filters"] = ValidationUtils.validate_json_string(
                filters, "filters"
            )
        else:
            validated_params["filters"] = None

        return validated_params

    @staticmethod
    def validate_spaces_list_params(
        space_type: str | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize spaces list parameters."""
        return {
            "type": ValidationUtils.validate_space_type(space_type),
            "sort": ValidationUtils.validate_space_sort_field(sort),
        }


class PydanticValidatorMixin:
    """Mixin providing Pydantic field validators using shared validation logic."""

    @classmethod
    def validate_query_field(cls, v: str) -> str:
        """Pydantic field validator for query fields."""
        return ValidationUtils.validate_query(v)

    @classmethod
    def validate_cql_field(cls, v: str) -> str:
        """Pydantic field validator for CQL fields."""
        return ValidationUtils.validate_cql_query(v)

    @classmethod
    def validate_content_type_field(cls, v: str | None) -> str | None:
        """Pydantic field validator for content_type fields."""
        return ValidationUtils.validate_content_type(v)

    @classmethod
    def validate_sort_by_field(cls, v: list[str] | None) -> list[str] | None:
        """Pydantic field validator for sort_by fields."""
        return ValidationUtils.validate_sort_fields(v)

    @classmethod
    def validate_sort_direction_field(cls, v: list[str] | None) -> list[str] | None:
        """Pydantic field validator for sort_direction fields."""
        return ValidationUtils.validate_sort_directions(v)

    @classmethod
    def validate_max_results_field(cls, v: int | None) -> int | None:
        """Pydantic field validator for max_results fields."""
        return ValidationUtils.validate_max_results(v)
