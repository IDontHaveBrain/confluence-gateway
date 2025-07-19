"""Unified exception-to-response mapping system for API and CLI interfaces."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    ConfluenceGatewayError,
    EmbeddingCompatibilityError,
    EmbeddingError,
    EmbeddingProviderError,
    GenerationError,
    SearchParameterError,
    SemanticSearchError,
)

logger = logging.getLogger(__name__)


@dataclass
class ErrorInfo:
    """Normalized error information for both API and CLI interfaces."""

    error_type: str
    status_code: int
    message: str
    details: dict[str, Any] | None = None
    severity: str = "error"  # error, warning, info


class ExceptionMapper:
    """Central exception mapping system supporting both API and CLI interfaces."""

    # Exception type to HTTP status code and severity mapping
    _EXCEPTION_MAPPING: dict[type[Exception], dict[str, Any]] = {
        # Authentication & Authorization
        ConfluenceAuthenticationError: {
            "status_code": 401,
            "severity": "error",
            "details_type": "authentication_error",
        },
        # Network & Connection
        ConfluenceConnectionError: {
            "status_code": 503,
            "severity": "error",
            "details_type": "connection_error",
        },
        # API Errors
        ConfluenceAPIError: {
            "status_code": None,  # Dynamic based on exception.status_code
            "severity": "error",
            "details_type": "api_error",
        },
        # Search & Query
        SearchParameterError: {
            "status_code": 400,
            "severity": "error",
            "details_type": "search_parameter_error",
        },
        SemanticSearchError: {
            "status_code": 500,
            "severity": "error",
            "details_type": "semantic_search_error",
        },
        # Generation
        GenerationError: {
            "status_code": 500,
            "severity": "error",
            "details_type": "generation_error",
        },
        # Embedding
        EmbeddingError: {
            "status_code": 500,
            "severity": "error",
            "details_type": "embedding_error",
        },
        EmbeddingProviderError: {
            "status_code": 500,
            "severity": "error",
            "details_type": "embedding_provider_error",
        },
        EmbeddingCompatibilityError: {
            "status_code": 400,
            "severity": "error",
            "details_type": "embedding_compatibility_error",
        },
        # Generic Gateway Error
        ConfluenceGatewayError: {
            "status_code": 500,
            "severity": "error",
            "details_type": "gateway_error",
        },
        # Generic Exception
        Exception: {
            "status_code": 500,
            "severity": "error",
            "details_type": "internal_error",
        },
    }

    @classmethod
    def map_exception(cls, exception: Exception) -> ErrorInfo:
        """Map an exception to normalized error information."""
        exception_type = type(exception)
        error_type_name = exception_type.__name__

        # Find the most specific mapping
        mapping = cls._get_exception_mapping(exception_type)

        # Determine status code
        status_code = cls._get_status_code(exception, mapping)

        # Build error details
        details = cls._build_error_details(exception, mapping)

        # Log the error for debugging
        logger.error(
            f"Exception mapped: {error_type_name} -> {status_code}", exc_info=True
        )

        return ErrorInfo(
            error_type=error_type_name,
            status_code=status_code,
            message=str(exception),
            details=details,
            severity=mapping["severity"],
        )

    @classmethod
    def _get_exception_mapping(cls, exception_type: type[Exception]) -> dict[str, Any]:
        """Get the most specific mapping for an exception type."""
        # Walk the MRO to find the first matching exception type
        for exc_type in exception_type.__mro__:
            if exc_type in cls._EXCEPTION_MAPPING:
                return cls._EXCEPTION_MAPPING[exc_type]

        # Fallback to generic Exception mapping
        return cls._EXCEPTION_MAPPING[Exception]

    @classmethod
    def _get_status_code(cls, exception: Exception, mapping: dict[str, Any]) -> int:
        """Determine the appropriate status code for an exception."""
        # Handle dynamic status codes (e.g., ConfluenceAPIError)
        if mapping["status_code"] is None:
            if isinstance(exception, ConfluenceAPIError) and exception.status_code:
                return exception.status_code
            return 500  # Default fallback

        return int(mapping["status_code"])

    @classmethod
    def _build_error_details(
        cls, exception: Exception, mapping: dict[str, Any]
    ) -> dict[str, Any]:
        """Build error details dictionary for an exception."""
        details = {"type": mapping["details_type"]}

        # Add exception-specific details
        if isinstance(exception, ConfluenceAPIError):
            if exception.status_code:
                details["api_status_code"] = exception.status_code
            if exception.error_message:
                details["api_error_message"] = exception.error_message

        elif isinstance(exception, ConfluenceConnectionError):
            if exception.cause:
                details["cause"] = str(exception.cause)
                details["cause_type"] = type(exception.cause).__name__

        return details


class APIExceptionHandler:
    """API-specific exception handling using the unified mapping system."""

    @staticmethod
    def create_http_exception(exception: Exception) -> dict[str, Any]:
        """Create FastAPI HTTPException detail from exception."""
        from confluence_gateway.api.schemas.responses import ErrorResponse

        error_info = ExceptionMapper.map_exception(exception)

        error_response = ErrorResponse(
            code=error_info.status_code,
            message=error_info.message,
            details=error_info.details,
        )

        return {
            "status_code": error_info.status_code,
            "detail": error_response.model_dump(),
        }

    @staticmethod
    def handle_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for API routes using unified exception mapping."""
        from functools import wraps

        from fastapi import HTTPException

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise FastAPI HTTPExceptions as-is
                raise
            except Exception as e:
                exception_data = APIExceptionHandler.create_http_exception(e)
                raise HTTPException(**exception_data)

        return wrapper


class CLIExceptionHandler:
    """CLI-specific exception handling using the unified mapping system."""

    @staticmethod
    def format_error_message(exception: Exception, verbose: bool = False) -> str:
        """Format exception for CLI output."""
        error_info = ExceptionMapper.map_exception(exception)

        # Basic error message
        message = f"Error ({error_info.error_type}): {error_info.message}"

        # Add details if verbose mode
        if verbose and error_info.details:
            details_str = ", ".join(f"{k}={v}" for k, v in error_info.details.items())
            message += f" [{details_str}]"

        return message

    @staticmethod
    def handle_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for CLI commands using unified exception mapping."""
        from functools import wraps

        import typer

        from confluence_gateway.cli.common import print_status

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except typer.Exit:
                # Re-raise typer.Exit as-is
                raise
            except Exception as e:
                error_message = CLIExceptionHandler.format_error_message(e)
                logger.error(f"CLI Error: {error_message}", exc_info=True)
                print_status(error_message, "error")
                raise typer.Exit(code=1)

        return wrapper

    @staticmethod
    def handle_exceptions_verbose(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for CLI commands with verbose error output."""
        from functools import wraps

        import typer

        from confluence_gateway.cli.common import print_status

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except typer.Exit:
                # Re-raise typer.Exit as-is
                raise
            except Exception as e:
                error_message = CLIExceptionHandler.format_error_message(
                    e, verbose=True
                )
                logger.error(f"CLI Error: {error_message}", exc_info=True)
                print_status(error_message, "error")
                print_status("Check logs for more details.", "dim")
                raise typer.Exit(code=1)

        return wrapper
