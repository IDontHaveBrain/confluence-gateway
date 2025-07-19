"""Mock context managers and factory patterns for Confluence Gateway tests.

This module consolidates 8+ mocking patterns found across integration tests into
reusable factory classes and context managers. It provides standardized mock
creation, configuration injection, and common test patterns.

Key consolidations:
- LiteLLM async mocking patterns (test_e2e_workflows.py, test_rag_integration.py)
- Environment context management (test_rag_integration.py)
- Shared embedding model injection (test_configuration_matrix.py)
- Sentence transformer mocking (test_embedding_providers.py)
- Configuration path management patterns
- Response validation and error handling

Usage:
    from tests.utils.mock_helpers import MockContextFactory, LiteLLMContext

    # Simple LiteLLM mocking
    with LiteLLMContext("Test response") as mock_litellm:
        # Run code that uses litellm.acompletion
        pass

    # Environment management
    with EnvironmentContext({"ENV_VAR": "value"}) as env:
        # Code runs with temporary environment
        pass

    # Complex mock factory usage
    factory = MockContextFactory()
    with factory.create_full_test_context(
        litellm_response="AI response",
        env_vars={"KEY": "value"},
        shared_model=True
    ) as context:
        # Comprehensive test setup
        pass
"""

import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from tests.fixtures.test_utils import create_mock_litellm_response


class MockContextFactory:
    """Factory for creating standardized mock contexts across integration tests.

    This factory consolidates common mocking patterns found in integration tests
    into reusable context managers and utilities.
    """

    def __init__(self):
        """Initialize the mock context factory."""
        self._active_contexts: list[Any] = []

    @contextmanager
    def create_litellm_context(
        self, response_content: str, mock_name: str = "mock_litellm"
    ) -> Generator[AsyncMock, None, None]:
        """Create LiteLLM async mock context.

        Consolidates pattern found in:
        - test_e2e_workflows.py (lines 379-384)
        - test_rag_integration.py (lines 86-90, 136-139, 191-194)

        Args:
            response_content: Content for the mock LiteLLM response
            mock_name: Name for the mock (for debugging)

        Yields:
            AsyncMock configured for LiteLLM acompletion

        Example:
            with factory.create_litellm_context("AI response") as mock_litellm:
                # litellm.acompletion calls will return mock response
                result = await litellm.acompletion(...)
        """
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
            mock_litellm.return_value = create_mock_litellm_response(response_content)
            mock_litellm._mock_name = mock_name
            self._active_contexts.append(mock_litellm)

            try:
                yield mock_litellm
            finally:
                if mock_litellm in self._active_contexts:
                    self._active_contexts.remove(mock_litellm)

    @contextmanager
    def create_environment_context(
        self, env_vars: dict[str, str], config_path: str | None = None
    ) -> Generator[dict[str, str], None, None]:
        """Create temporary environment variable context.

        Consolidates pattern found in:
        - test_rag_integration.py (lines 34-62)
        - Multiple files with apply_env_vars/restore_env_vars patterns

        Args:
            env_vars: Environment variables to set temporarily
            config_path: Optional config file path to set

        Yields:
            Dictionary of applied environment variables

        Example:
            with factory.create_environment_context({"API_KEY": "test"}) as env:
                # Code runs with temporary environment variables
                pass
        """
        original_env = os.environ.copy()

        try:
            # Apply new environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            # Add config path if provided
            if config_path:
                os.environ["CONFLUENCE_GATEWAY_CONFIG_PATH"] = config_path

            yield env_vars
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    @contextmanager
    def create_sentence_transformer_context(
        self, shared_model: Any | None = None, mock_when_none: bool = True
    ) -> Generator[Any, None, None]:
        """Create sentence transformer mock context.

        Consolidates pattern found in:
        - test_configuration_matrix.py (lines 86-89, 167-174, 261-267)
        - test_embedding_providers.py (lines 52-79)
        - tests/fixtures/shared_embedding.py (lines 320-331)

        Args:
            shared_model: Pre-loaded model to inject, or None for mock
            mock_when_none: Whether to create a mock when shared_model is None

        Yields:
            Model instance (real shared model or mock)

        Example:
            with factory.create_sentence_transformer_context(shared_model) as model:
                # SentenceTransformer constructor returns the shared model
                pass
        """
        if shared_model is not None:
            # Use real shared model
            with patch(
                "sentence_transformers.SentenceTransformer",
                return_value=shared_model,
            ):
                yield shared_model
        elif mock_when_none:
            # Create mock model
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1] * 384]  # Mock embedding
            mock_model.get_sentence_embedding_dimension.return_value = 384

            with patch(
                "sentence_transformers.SentenceTransformer", return_value=mock_model
            ):
                yield mock_model
        else:
            # No-op context
            with patch("os.environ", os.environ):  # No-op patch
                yield None

    @contextmanager
    def create_full_test_context(
        self,
        litellm_response: str | None = None,
        env_vars: dict[str, str] | None = None,
        config_path: str | None = None,
        shared_model: Any | None = None,
        mock_sentence_transformers: bool = True,
    ) -> Generator[dict[str, Any], None, None]:
        """Create comprehensive test context with multiple mocks.

        Combines common patterns for complete test setup with all necessary mocks.

        Args:
            litellm_response: Content for LiteLLM mock response
            env_vars: Environment variables to set
            config_path: Config file path to set
            shared_model: Shared sentence transformer model
            mock_sentence_transformers: Whether to mock sentence transformers

        Yields:
            Dictionary containing all created mocks and contexts

        Example:
            with factory.create_full_test_context(
                litellm_response="Test AI response",
                env_vars={"GENERATION_ENABLE": "true"},
                shared_model=shared_sentence_transformer_model
            ) as context:
                # All mocks are active
                mock_litellm = context["litellm_mock"]
                model = context["sentence_transformer_model"]
        """
        context = {}

        # Stack context managers
        contexts = []

        try:
            # Environment context
            if env_vars:
                env_context = self.create_environment_context(env_vars, config_path)
                env_vars_applied = env_context.__enter__()
                contexts.append(env_context)
                context["env_vars"] = env_vars_applied

            # Sentence transformer context
            if mock_sentence_transformers:
                st_context = self.create_sentence_transformer_context(shared_model)
                st_model = st_context.__enter__()
                contexts.append(st_context)
                context["sentence_transformer_model"] = st_model

            # LiteLLM context
            if litellm_response:
                litellm_context = self.create_litellm_context(litellm_response)
                litellm_mock = litellm_context.__enter__()
                contexts.append(litellm_context)
                context["litellm_mock"] = litellm_mock

            yield context

        finally:
            # Clean up contexts in reverse order
            for ctx in reversed(contexts):
                try:
                    ctx.__exit__(None, None, None)
                except Exception as e:
                    print(f"Warning: Error cleaning up context: {e}")


# Convenience context managers for common patterns
@contextmanager
def LiteLLMContext(response_content: str) -> Generator[AsyncMock, None, None]:
    """Convenience context manager for LiteLLM mocking.

    Args:
        response_content: Content for the mock response

    Yields:
        AsyncMock configured for LiteLLM

    Example:
        with LiteLLMContext("AI response") as mock_litellm:
            # litellm.acompletion calls will return mock response
            pass
    """
    factory = MockContextFactory()
    with factory.create_litellm_context(response_content) as mock:
        yield mock


@contextmanager
def EnvironmentContext(
    env_vars: dict[str, str], config_path: str | None = None
) -> Generator[dict[str, str], None, None]:
    """Convenience context manager for environment variable management.

    Args:
        env_vars: Environment variables to set
        config_path: Optional config file path

    Yields:
        Dictionary of applied environment variables

    Example:
        with EnvironmentContext({"API_KEY": "test"}) as env:
            # Temporary environment is active
            pass
    """
    factory = MockContextFactory()
    with factory.create_environment_context(env_vars, config_path) as env:
        yield env


@contextmanager
def SentenceTransformerContext(
    shared_model: Any | None = None,
) -> Generator[Any, None, None]:
    """Convenience context manager for sentence transformer mocking.

    Args:
        shared_model: Pre-loaded model to inject

    Yields:
        Model instance (real or mock)

    Example:
        with SentenceTransformerContext(shared_model) as model:
            # SentenceTransformer constructor returns the model
            pass
    """
    factory = MockContextFactory()
    with factory.create_sentence_transformer_context(shared_model) as model:
        yield model


class ConfigurationMockHelper:
    """Helper for configuration-related mocking patterns.

    Consolidates configuration injection and temporary path management
    patterns found across tests.
    """

    @staticmethod
    @contextmanager
    def temporary_config_path(
        config_content: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Create temporary configuration file and path.

        Args:
            config_content: Optional config data to write to file

        Yields:
            Path to temporary config file

        Example:
            with ConfigurationMockHelper.temporary_config_path() as config_path:
                # Use config_path in test
                pass
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            if config_content:
                import json

                json.dump(config_content, temp_file, indent=2)
            temp_path = temp_file.name

        try:
            yield temp_path
        finally:
            try:
                Path(temp_path).unlink()
            except OSError:
                pass  # File might already be deleted

    @staticmethod
    @contextmanager
    def mock_confluence_credentials() -> Generator[tuple[str, str, str], None, None]:
        """Mock Confluence credentials for testing.

        Yields:
            Tuple of (url, username, api_token)

        Example:
            with ConfigurationMockHelper.mock_confluence_credentials() as (url, user, token):
                # Use mock credentials
                pass
        """
        mock_url = "https://test.atlassian.net"
        mock_username = "test@example.com"
        mock_token = "test_api_token"

        with patch.dict(
            os.environ,
            {
                "CONFLUENCE_URL": mock_url,
                "CONFLUENCE_USERNAME": mock_username,
                "CONFLUENCE_API_TOKEN": mock_token,
            },
        ):
            yield (mock_url, mock_username, mock_token)


class ResponseValidationHelper:
    """Helper for response validation and assertion patterns.

    Consolidates response validation patterns found across API and CLI tests.
    """

    @staticmethod
    def validate_cli_json_response(
        response_data: dict[str, Any],
        required_keys: list[str],
        response_name: str = "response",
    ) -> None:
        """Validate CLI JSON response structure.

        Args:
            response_data: Parsed JSON response data
            required_keys: Keys that must be present
            response_name: Name for error messages

        Raises:
            AssertionError: If validation fails

        Example:
            validate_cli_json_response(
                data,
                ["results", "total", "took_ms"],
                "search response"
            )
        """
        for key in required_keys:
            assert key in response_data, (
                f"{response_name} missing required key: {key}. "
                f"Available keys: {list(response_data.keys())}"
            )

    @staticmethod
    def validate_api_response(
        response: Any,
        expected_status: int = 200,
        required_keys: list[str] | None = None,
        response_name: str = "API response",
    ) -> dict[str, Any]:
        """Validate API response structure and return data.

        Args:
            response: HTTP response object
            expected_status: Expected status code
            required_keys: Keys that must be present in JSON
            response_name: Name for error messages

        Returns:
            Parsed JSON response data

        Raises:
            AssertionError: If validation fails

        Example:
            data = validate_api_response(
                response,
                200,
                ["results", "total"],
                "search API"
            )
        """
        assert response.status_code == expected_status, (
            f"{response_name} returned status {response.status_code}, "
            f"expected {expected_status}. Response: {response.text}"
        )

        data = response.json()

        if required_keys:
            for key in required_keys:
                assert key in data, (
                    f"{response_name} missing required key: {key}. "
                    f"Available keys: {list(data.keys())}"
                )

        return data


class PerformanceMockHelper:
    """Helper for performance-related mocking and measurement.

    Integrates with the performance testing utilities while providing
    mock-specific timing and validation capabilities.
    """

    @staticmethod
    @contextmanager
    def timed_mock_context(
        mock_name: str, performance_threshold: float | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Create timed mock context with performance tracking.

        Args:
            mock_name: Name for performance logging
            performance_threshold: Optional time threshold in seconds

        Yields:
            Dictionary with timing information

        Example:
            with PerformanceMockHelper.timed_mock_context("test_operation", 1.0) as timer:
                # Timed operation
                pass
            # timer["duration"] contains elapsed time
        """
        start_time = time.time()
        timing_info = {"start_time": start_time}

        try:
            yield timing_info
        finally:
            end_time = time.time()
            duration = end_time - start_time
            timing_info["end_time"] = end_time
            timing_info["duration"] = duration

            # Log to existing performance system
            from tests.utils.performance_helpers import log_embedding_operation

            log_embedding_operation(mock_name, duration)

            # Check threshold if specified
            if performance_threshold and duration > performance_threshold:
                raise AssertionError(
                    f"Mock operation '{mock_name}' exceeded threshold: "
                    f"{duration:.3f}s > {performance_threshold:.3f}s"
                )


# Migration utilities for transitioning existing code
class MockMigrationHelper:
    """Helper for migrating existing mock patterns to the new utilities.

    Provides transition utilities and compatibility functions for updating
    existing test code to use the consolidated mock helpers.
    """

    @staticmethod
    def migrate_litellm_pattern(response_content: str) -> Any:
        """Migrate old LiteLLM mocking pattern to new helper.

        OLD PATTERN:
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
                mock_litellm.return_value = create_mock_litellm_response(content)
                # test code

        NEW PATTERN:
            with LiteLLMContext(content) as mock_litellm:
                # test code

        Args:
            response_content: Content for mock response

        Returns:
            Context manager for the mock
        """
        return LiteLLMContext(response_content)

    @staticmethod
    def migrate_environment_pattern(env_vars: dict[str, str]) -> Any:
        """Migrate old environment variable pattern to new helper.

        OLD PATTERN:
            original_env = os.environ.copy()
            try:
                for key, value in env_vars.items():
                    os.environ[key] = value
                # test code
            finally:
                os.environ.clear()
                os.environ.update(original_env)

        NEW PATTERN:
            with EnvironmentContext(env_vars):
                # test code

        Args:
            env_vars: Environment variables to set

        Returns:
            Context manager for environment
        """
        return EnvironmentContext(env_vars)

    @staticmethod
    def show_migration_examples() -> None:
        """Print migration examples for common patterns."""
        print("MOCK HELPERS MIGRATION EXAMPLES")
        print("=" * 50)

        print("\n1. LiteLLM Mocking:")
        print("OLD:")
        print("""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
            mock_litellm.return_value = create_mock_litellm_response("response")
            # test code
        """)

        print("NEW:")
        print("""
        with LiteLLMContext("response") as mock_litellm:
            # test code
        """)

        print("\n2. Environment Management:")
        print("OLD:")
        print("""
        original_env = os.environ.copy()
        try:
            os.environ.update({"KEY": "value"})
            # test code
        finally:
            os.environ.clear()
            os.environ.update(original_env)
        """)

        print("NEW:")
        print("""
        with EnvironmentContext({"KEY": "value"}):
            # test code
        """)

        print("\n3. Combined Context:")
        print("OLD:")
        print("""
        # Multiple nested context managers
        with env_context:
            with sentence_transformer_context:
                with litellm_context:
                    # test code
        """)

        print("NEW:")
        print("""
        factory = MockContextFactory()
        with factory.create_full_test_context(
            litellm_response="response",
            env_vars={"KEY": "value"},
            shared_model=model
        ) as context:
            # test code
        """)


# Export key classes and functions
__all__ = [
    "MockContextFactory",
    "LiteLLMContext",
    "EnvironmentContext",
    "SentenceTransformerContext",
    "ConfigurationMockHelper",
    "ResponseValidationHelper",
    "PerformanceMockHelper",
    "MockMigrationHelper",
]
