"""SearchValidator service for handling input validation and sanitization operations."""

import logging
import re
from typing import Any

from confluence_gateway.core.config import SearchConfig, get_search_config
from confluence_gateway.core.exceptions import SearchParameterError

logger = logging.getLogger(__name__)


class SearchValidator:
    """Service for validating and sanitizing search inputs."""

    def __init__(self) -> None:
        """Initialize SearchValidator with configuration."""
        self.config: SearchConfig = get_search_config()
        logger.info("SearchValidator initialized successfully.")

    def sanitize_keywords(self, keywords: str | list[str]) -> str:
        """
        Sanitize search keywords by cleaning and validating input.

        Args:
            keywords: String or list of strings to sanitize

        Returns:
            Sanitized keywords as a single string

        Raises:
            SearchParameterError: If keywords are empty or invalid
        """
        if isinstance(keywords, str):
            return self.sanitize_text(keywords)

        if not keywords:
            raise SearchParameterError("Search keywords cannot be empty")

        sanitized_keywords = [self.sanitize_text(kw) for kw in keywords]
        return " ".join(sanitized_keywords)

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by cleaning and validating.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text

        Raises:
            SearchParameterError: If text is empty or invalid
        """
        if not text:
            raise SearchParameterError("Search text cannot be empty")

        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", text.strip())

        if not sanitized:
            raise SearchParameterError("Search text cannot be empty")

        if len(sanitized) < 2:
            raise SearchParameterError("Search text must be at least 2 characters long")

        # Remove potentially dangerous characters while preserving necessary ones
        sanitized = re.sub(
            r"[^\w\s\-.,;:!?\'\"()/+*=%&#@$^~]", "", sanitized, flags=re.UNICODE
        )

        return sanitized

    def validate_cql_query(self, cql: str) -> str:
        """
        Validate CQL (Confluence Query Language) query structure.

        Args:
            cql: CQL query string to validate

        Returns:
            The validated CQL query (trimmed)

        Raises:
            SearchParameterError: If CQL query is invalid
        """
        if not cql or not cql.strip():
            raise SearchParameterError("CQL query cannot be empty")

        # Define valid CQL operators and patterns
        equality_operators = ["=", "!=", "~", "^=", "$=", "*="]
        comparison_operators = ["<", ">", "<=", ">="]
        logical_operators = ["AND", "OR", "NOT"]

        # Pattern to match field-operator-value combinations
        field_operator_value_pattern = re.compile(
            r"\b[\w.-]+\s*("
            + "|".join(map(re.escape, equality_operators + comparison_operators))
            + r')\s*("([^"]|\\")*"|\'([^\']|\\\')*\'|\S+)',
            re.IGNORECASE,
        )

        # Pattern to match logical operators with conditions
        logical_pattern = re.compile(
            r"\b("
            + "|".join(map(re.escape, logical_operators))
            + r")\b\s+"
            + r"("
            + r"\(|"
            + r"\b[\w.-]+\b\s*("
            + "|".join(map(re.escape, equality_operators + comparison_operators))
            + r")"
            + r")",
            re.IGNORECASE,
        )

        # Pattern to match ORDER BY clauses
        order_by_pattern = re.compile(
            r"\bORDER\s+BY\s+[\w.-]+(\s+(ASC|DESC))?", re.IGNORECASE
        )

        # Check for valid CQL structure
        has_field_operator_value = bool(field_operator_value_pattern.search(cql))
        has_logical_structure = bool(logical_pattern.search(cql))
        has_order_by = bool(order_by_pattern.search(cql))
        has_parentheses = "(" in cql and ")" in cql

        if not (
            has_field_operator_value
            or has_logical_structure
            or has_order_by
            or has_parentheses
        ):
            raise SearchParameterError(
                "Invalid CQL query format. CQL must contain field-operator-value patterns, "
                "logical operators with conditions, ORDER BY clauses, or parentheses."
            )

        return cql.strip()

    def validate_limit_parameter(self, limit: int | None) -> int:
        """
        Validate and normalize limit parameter.

        Args:
            limit: Limit value to validate

        Returns:
            Validated limit value

        Raises:
            SearchParameterError: If limit is invalid
        """
        if limit is not None:
            if limit <= 0 or limit > self.config.max_limit:
                raise SearchParameterError(
                    f"Limit must be between 1 and {self.config.max_limit}"
                )
            return limit
        else:
            return self.config.default_limit

    def validate_start_parameter(self, start: int | None) -> int:
        """
        Validate and normalize start parameter.

        Args:
            start: Start value to validate

        Returns:
            Validated start value

        Raises:
            SearchParameterError: If start is invalid
        """
        if start is not None and start < 0:
            raise SearchParameterError("Start position cannot be negative")
        return start if start is not None else 0

    def validate_top_k_parameter(self, top_k: int | None) -> None:
        """
        Validate top_k parameter for semantic search.

        Args:
            top_k: Top-k value to validate

        Raises:
            SearchParameterError: If top_k is invalid
        """
        if top_k is not None and top_k <= 0:
            raise SearchParameterError("top_k must be a positive integer")

    def validate_min_relevance_parameter(self, min_relevance: float | None) -> None:
        """
        Validate min_relevance parameter.

        Args:
            min_relevance: Min relevance value to validate

        Raises:
            SearchParameterError: If min_relevance is invalid
        """
        if min_relevance is not None and (min_relevance < 0.0 or min_relevance > 1.0):
            raise SearchParameterError("min_relevance must be between 0.0 and 1.0")

    def validate_query_parameter(self, query: str | None) -> None:
        """
        Validate query parameter for semantic search.

        Args:
            query: Query string to validate

        Raises:
            SearchParameterError: If query is invalid
        """
        if query is not None and (not query or query.isspace()):
            raise SearchParameterError("Search query cannot be empty")

    def validate_semantic_search_parameters(
        self, query: str, top_k: int, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Centralized validation for semantic search parameters.

        Args:
            query: Search query string
            top_k: Number of top results to return
            **kwargs: Additional parameters to validate

        Returns:
            Dictionary of validated parameters

        Raises:
            SearchParameterError: If parameters are invalid
        """
        # Validate query parameter
        if not query or query.isspace():
            raise SearchParameterError("Semantic search query cannot be empty.")

        # Validate top_k parameter
        if top_k <= 0:
            raise SearchParameterError("top_k must be a positive integer.")

        # Sanitize query
        sanitized_query = query.strip()

        # Build validated parameters
        validated_params = dict(kwargs)
        validated_params["query"] = sanitized_query
        validated_params["top_k"] = top_k

        return validated_params

    def validate_search_parameters(self, **kwargs: Any) -> dict[str, Any]:
        """
        Validate common search parameters using centralized validation methods.

        Args:
            **kwargs: Search parameters to validate

        Returns:
            Dictionary of validated parameters

        Raises:
            SearchParameterError: If parameters are invalid
        """
        validated_params = {}

        # Validate limit parameter
        validated_params["limit"] = self.validate_limit_parameter(kwargs.get("limit"))

        # Validate start parameter
        validated_params["start"] = self.validate_start_parameter(kwargs.get("start"))

        # Validate top_k parameter for semantic search
        self.validate_top_k_parameter(kwargs.get("top_k"))

        # Validate min_relevance parameter
        self.validate_min_relevance_parameter(kwargs.get("min_relevance"))

        # Validate query parameter for semantic search
        self.validate_query_parameter(kwargs.get("query"))

        # Copy other parameters as-is
        for key, value in kwargs.items():
            if key not in validated_params:
                validated_params[key] = value

        return validated_params

    def validate_and_normalize_search_params(
        self, limit: int | None = None, start: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Centralized method to validate and normalize search parameters.
        This method replaces the validate_search_params decorator functionality.

        Args:
            limit: Maximum number of results to return
            start: Start position for pagination
            **kwargs: Additional parameters to pass through

        Returns:
            Dictionary with validated limit, start, and other parameters

        Raises:
            SearchParameterError: If parameters are invalid
        """
        validated_limit = self.validate_limit_parameter(limit)
        validated_start = self.validate_start_parameter(start)

        # Build result with validated parameters
        result = dict(kwargs)
        result["limit"] = validated_limit
        result["start"] = validated_start

        return result
