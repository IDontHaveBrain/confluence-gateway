"""Unified error handling decorators for service layer operations.

This module provides decorators to consolidate the repetitive exception handling
patterns found across service files. It supports both generic and specialized
error handling for common service operations.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceConnectionError,
    EmbeddingError,
    EmbeddingProviderError,
    GenerationError,
    SemanticSearchError,
)

T = TypeVar("T", bound=Callable[..., Any])


class ServiceErrorHandler:
    """Generic service error handler decorator class."""

    def __init__(
        self,
        operation_name: str,
        target_exception: type[Exception],
        logger: logging.Logger,
        specific_mappings: dict[type[BaseException], str] | None = None,
        generic_error_message: str | None = None,
    ):
        """Initialize the error handler decorator.

        Args:
            operation_name: Name of the operation for error messages
            target_exception: Exception type to raise on errors
            logger: Logger instance for error logging
            specific_mappings: Dict mapping specific exception types to error messages
            generic_error_message: Custom message for unexpected errors
        """
        self.operation_name = operation_name
        self.target_exception = target_exception
        self.logger = logger
        self.specific_mappings = specific_mappings or {}
        self.generic_error_message = (
            generic_error_message
            or f"An unexpected error occurred during {operation_name}"
        )

    def __call__(self, func: T) -> T:
        """Apply the error handling decorator to a function."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except tuple(self.specific_mappings.keys()) as e:
                error_message = self.specific_mappings[type(e)]
                self.logger.error(f"{error_message}: {e}", exc_info=True)
                raise self.target_exception(error_message) from e
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during {self.operation_name}: {e}",
                    exc_info=True,
                )
                raise self.target_exception(f"{self.generic_error_message}: {e}") from e

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except tuple(self.specific_mappings.keys()) as e:
                error_message = self.specific_mappings[type(e)]
                self.logger.error(f"{error_message}: {e}", exc_info=True)
                raise self.target_exception(error_message) from e
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during {self.operation_name}: {e}",
                    exc_info=True,
                )
                raise self.target_exception(f"{self.generic_error_message}: {e}") from e

        # Return the appropriate wrapper based on whether the function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]


def service_error_handler(
    operation_name: str,
    target_exception: type[Exception],
    logger: logging.Logger,
    specific_mappings: dict[type[BaseException], str] | None = None,
    generic_error_message: str | None = None,
) -> Callable[[T], T]:
    """Generic service error handler decorator.

    Args:
        operation_name: Name of the operation for error messages
        target_exception: Exception type to raise on errors
        logger: Logger instance for error logging
        specific_mappings: Dict mapping specific exception types to error messages
        generic_error_message: Custom message for unexpected errors

    Returns:
        Decorator function
    """
    return ServiceErrorHandler(
        operation_name=operation_name,
        target_exception=target_exception,
        logger=logger,
        specific_mappings=specific_mappings,
        generic_error_message=generic_error_message,
    )


def embedding_error_handler(
    logger: logging.Logger, operation_name: str = "embedding operation"
) -> Callable[[T], T]:
    """Specialized error handler for embedding operations.

    Handles the common pattern of catching EmbeddingProviderError and
    re-raising as EmbeddingError.

    Args:
        logger: Logger instance for error logging
        operation_name: Specific operation name for error messages

    Returns:
        Decorator function
    """
    specific_mappings: dict[type[BaseException], str] = {
        EmbeddingProviderError: f"Embedding provider failed during {operation_name}",
    }

    return ServiceErrorHandler(
        operation_name=operation_name,
        target_exception=EmbeddingError,
        logger=logger,
        specific_mappings=specific_mappings,
        generic_error_message=f"Failed to complete {operation_name} due to provider error",
    )


def search_error_handler(
    logger: logging.Logger, operation_name: str = "search operation"
) -> Callable[[T], T]:
    """Specialized error handler for search operations.

    Handles the common pattern of catching EmbeddingError and
    re-raising as SemanticSearchError.

    Args:
        logger: Logger instance for error logging
        operation_name: Specific operation name for error messages

    Returns:
        Decorator function
    """
    specific_mappings: dict[type[BaseException], str] = {
        EmbeddingError: f"Embedding failed during {operation_name}",
    }

    return ServiceErrorHandler(
        operation_name=operation_name,
        target_exception=SemanticSearchError,
        logger=logger,
        specific_mappings=specific_mappings,
        generic_error_message=f"Search failed during {operation_name}",
    )


def generation_error_handler(
    logger: logging.Logger, operation_name: str = "generation operation"
) -> Callable[[T], T]:
    """Specialized error handler for generation operations.

    Handles the common pattern of catching SemanticSearchError and
    re-raising as GenerationError.

    Args:
        logger: Logger instance for error logging
        operation_name: Specific operation name for error messages

    Returns:
        Decorator function
    """
    specific_mappings: dict[type[BaseException], str] = {
        SemanticSearchError: f"Search failed during {operation_name}",
    }

    return ServiceErrorHandler(
        operation_name=operation_name,
        target_exception=GenerationError,
        logger=logger,
        specific_mappings=specific_mappings,
        generic_error_message=f"Generation failed during {operation_name}",
    )


def confluence_error_handler(
    logger: logging.Logger,
    operation_name: str = "confluence operation",
    continue_on_error: bool = False,
) -> Callable[[T], T]:
    """Specialized error handler for Confluence API operations.

    Handles the common pattern of catching Confluence-specific errors.
    Can be configured to log and continue rather than re-raise.

    Args:
        logger: Logger instance for error logging
        operation_name: Specific operation name for error messages
        continue_on_error: If True, logs errors and returns None instead of re-raising

    Returns:
        Decorator function
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except (ConfluenceAPIError, ConfluenceConnectionError) as e:
                error_msg = f"Confluence API error during {operation_name}: {e}"
                logger.error(error_msg, exc_info=True)
                if continue_on_error:
                    return None
                raise
            except Exception as e:
                error_msg = f"Unexpected error during {operation_name}: {e}"
                logger.error(error_msg, exc_info=True)
                if continue_on_error:
                    return None
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (ConfluenceAPIError, ConfluenceConnectionError) as e:
                error_msg = f"Confluence API error during {operation_name}: {e}"
                logger.error(error_msg, exc_info=True)
                if continue_on_error:
                    return None
                raise
            except Exception as e:
                error_msg = f"Unexpected error during {operation_name}: {e}"
                logger.error(error_msg, exc_info=True)
                if continue_on_error:
                    return None
                raise

        # Return the appropriate wrapper based on whether the function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator


def litellm_error_handler(
    logger: logging.Logger,
    model_name: str,
    operation_name: str = "LLM operation",
) -> Callable[[T], T]:
    """Specialized error handler for LiteLLM operations.

    Handles the common pattern of catching LiteLLM exceptions and
    re-raising as GenerationError.

    Args:
        logger: Logger instance for error logging
        model_name: Name of the LLM model being used
        operation_name: Specific operation name for error messages

    Returns:
        Decorator function
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if it's a LiteLLM exception by attempting to get LiteLLM exceptions
                try:
                    from confluence_gateway.services.generation import (
                        _get_litellm_exceptions,
                    )

                    litellm_exceptions = tuple(_get_litellm_exceptions().values())
                    if isinstance(e, litellm_exceptions):
                        error_type = type(e).__name__
                        logger.error(
                            f"LiteLLM API error calling model '{model_name}': {error_type}: {e}",
                            exc_info=True,
                        )
                        raise GenerationError(
                            f"LLM API error ({error_type}): {e}"
                        ) from e
                except ImportError:
                    pass

                # Handle as generic error
                logger.error(
                    f"Unexpected error calling LLM model '{model_name}': {e}",
                    exc_info=True,
                )
                raise GenerationError(
                    f"An unexpected error occurred during {operation_name}: {e}"
                ) from e

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if it's a LiteLLM exception by attempting to get LiteLLM exceptions
                try:
                    from confluence_gateway.services.generation import (
                        _get_litellm_exceptions,
                    )

                    litellm_exceptions = tuple(_get_litellm_exceptions().values())
                    if isinstance(e, litellm_exceptions):
                        error_type = type(e).__name__
                        logger.error(
                            f"LiteLLM API error calling model '{model_name}': {error_type}: {e}",
                            exc_info=True,
                        )
                        raise GenerationError(
                            f"LLM API error ({error_type}): {e}"
                        ) from e
                except ImportError:
                    pass

                # Handle as generic error
                logger.error(
                    f"Unexpected error calling LLM model '{model_name}': {e}",
                    exc_info=True,
                )
                raise GenerationError(
                    f"An unexpected error occurred during {operation_name}: {e}"
                ) from e

        # Return the appropriate wrapper based on whether the function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator


# Convenience functions for common patterns
def embed_text_error_handler(logger: logging.Logger) -> Callable[[T], T]:
    """Convenience decorator for text embedding operations."""
    return embedding_error_handler(logger, "text embedding")


def embed_texts_error_handler(logger: logging.Logger) -> Callable[[T], T]:
    """Convenience decorator for batch text embedding operations."""
    return embedding_error_handler(logger, "batch text embedding")


def semantic_search_error_handler(logger: logging.Logger) -> Callable[[T], T]:
    """Convenience decorator for semantic search operations."""
    return search_error_handler(logger, "semantic search")


def query_embedding_error_handler(logger: logging.Logger) -> Callable[[T], T]:
    """Convenience decorator for query embedding operations."""
    return search_error_handler(logger, "query embedding")


def context_retrieval_error_handler(logger: logging.Logger) -> Callable[[T], T]:
    """Convenience decorator for RAG context retrieval operations."""
    return generation_error_handler(logger, "context retrieval")


def llm_call_error_handler(logger: logging.Logger, model_name: str) -> Callable[[T], T]:
    """Convenience decorator for LLM API calls."""
    return litellm_error_handler(logger, model_name, "LLM call")
