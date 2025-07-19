"""Environment management utilities for integration tests.

This module provides context managers to consolidate the repetitive try/finally
environment setup/teardown patterns found throughout integration tests.

Example usage:
    # Basic pattern (replaces 17+ manual try/finally blocks)
    with EnvironmentManager.from_config(get_qdrant_memory_config()):
        # Test logic here - environment automatically restored on exit
        result = subprocess.run([...])
        assert result.returncode == 0

    # With custom environment variables
    with EnvironmentManager.from_config(
        get_qdrant_memory_config(),
        extra_env={"GENERATION_ENABLE": "true"}
    ):
        # Test logic with additional environment variables
        pass

    # Convenience methods for common scenarios
    with EnvironmentManager.qdrant_memory():
        # Qdrant memory configuration
        pass

    with EnvironmentManager.chroma_memory():
        # ChromaDB memory configuration
        pass

    with EnvironmentManager.no_vector_db():
        # Text-only configuration
        pass
"""

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

from tests.fixtures.config_builders import (
    ConfigBuilderResult,
    apply_env_vars,
    cleanup_temp_dirs,
    get_chroma_memory_config,
    get_no_vector_db_config,
    get_qdrant_memory_config,
    restore_env_vars,
)


class EnvironmentManager:
    """Context manager for automated environment setup and cleanup in tests.

    This class consolidates the repetitive try/finally environment management
    patterns found throughout integration tests, providing thread-safe cleanup
    and proper error handling.

    Attributes:
        config_result: The configuration result containing environment variables
                      and temporary directories to manage
        extra_env: Additional environment variables to apply
        previous_env: Store for environment variable restoration
        _lock: Thread-safety lock for environment modifications
    """

    def __init__(
        self,
        config_result: ConfigBuilderResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Initialize the environment manager.

        Args:
            config_result: Configuration result from config builders
            extra_env: Optional additional environment variables to apply
        """
        self.config_result = config_result
        self.extra_env = extra_env or {}
        self.previous_env: dict[str, str | None] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> "EnvironmentManager":
        """Enter the context manager and apply environment configuration."""
        with self._lock:
            # Combine base environment variables with extras
            combined_env = self.config_result.env_vars.copy()
            combined_env.update(self.extra_env)

            # Apply environment variables and store previous values
            self.previous_env = apply_env_vars(combined_env)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and restore environment state.

        Performs cleanup regardless of whether an exception occurred.
        Thread-safe restoration of environment variables and cleanup of
        temporary directories.
        """
        with self._lock:
            try:
                # Restore environment variables
                restore_env_vars(self.previous_env)
            finally:
                # Always cleanup temporary directories, even if restore fails
                cleanup_temp_dirs(self.config_result.temp_dirs)

    @classmethod
    def from_config(
        cls,
        config_result: ConfigBuilderResult,
        extra_env: dict[str, str] | None = None,
    ) -> "EnvironmentManager":
        """Create an EnvironmentManager from a configuration result.

        Args:
            config_result: Configuration result from config builders
            extra_env: Optional additional environment variables to apply

        Returns:
            EnvironmentManager instance ready for use as context manager

        Example:
            with EnvironmentManager.from_config(get_qdrant_memory_config()):
                # Test logic here
                pass
        """
        return cls(config_result, extra_env)

    @classmethod
    def qdrant_memory(
        cls,
        shared_provider: Any = None,
        extra_env: dict[str, str] | None = None,
    ) -> "EnvironmentManager":
        """Create an EnvironmentManager with Qdrant memory configuration.

        Args:
            shared_provider: Optional shared sentence-transformers provider
            extra_env: Optional additional environment variables to apply

        Returns:
            EnvironmentManager configured for Qdrant memory mode

        Example:
            with EnvironmentManager.qdrant_memory():
                # Test with Qdrant memory configuration
                pass
        """
        config_result = get_qdrant_memory_config(shared_provider=shared_provider)
        return cls(config_result, extra_env)

    @classmethod
    def chroma_memory(
        cls,
        shared_provider: Any = None,
        extra_env: dict[str, str] | None = None,
    ) -> "EnvironmentManager":
        """Create an EnvironmentManager with ChromaDB memory configuration.

        Args:
            shared_provider: Optional shared sentence-transformers provider
            extra_env: Optional additional environment variables to apply

        Returns:
            EnvironmentManager configured for ChromaDB memory mode

        Example:
            with EnvironmentManager.chroma_memory():
                # Test with ChromaDB memory configuration
                pass
        """
        config_result = get_chroma_memory_config(shared_provider=shared_provider)
        return cls(config_result, extra_env)

    @classmethod
    def no_vector_db(
        cls,
        extra_env: dict[str, str] | None = None,
    ) -> "EnvironmentManager":
        """Create an EnvironmentManager with text-only configuration.

        Args:
            extra_env: Optional additional environment variables to apply

        Returns:
            EnvironmentManager configured for text-only mode

        Example:
            with EnvironmentManager.no_vector_db():
                # Test with text-only configuration (no vector database)
                pass
        """
        config_result = get_no_vector_db_config()
        return cls(config_result, extra_env)

    @classmethod
    def rag_enabled(
        cls,
        base_config: ConfigBuilderResult | None = None,
        model_name: str = "openrouter/google/gemini-2.5-flash",
        api_key: str = "test_api_key",
        extra_env: dict[str, str] | None = None,
    ) -> "EnvironmentManager":
        """Create an EnvironmentManager with RAG generation enabled.

        Args:
            base_config: Base configuration (defaults to Qdrant memory)
            model_name: Generation model name
            api_key: LiteLLM API key for generation
            extra_env: Optional additional environment variables to apply

        Returns:
            EnvironmentManager configured for RAG generation testing

        Example:
            with EnvironmentManager.rag_enabled():
                # Test with RAG generation capabilities
                pass
        """
        if base_config is None:
            base_config = get_qdrant_memory_config()

        # Add RAG-specific environment variables
        rag_env = {
            "GENERATION_ENABLE": "true",
            "GENERATION_MODEL_NAME": model_name,
            "GENERATION_LITELLM_API_KEY": api_key,
        }

        # Combine with any additional environment variables
        combined_extra_env = rag_env.copy()
        if extra_env:
            combined_extra_env.update(extra_env)

        return cls(base_config, combined_extra_env)

    def get_env_var(self, key: str) -> str | None:
        """Get an environment variable value that was set by this manager.

        Args:
            key: Environment variable key

        Returns:
            Environment variable value or None if not set
        """
        combined_env = self.config_result.env_vars.copy()
        combined_env.update(self.extra_env)
        return combined_env.get(key)

    def get_temp_dirs(self) -> list[Path]:
        """Get list of temporary directories managed by this instance.

        Returns:
            List of temporary directory paths that will be cleaned up
        """
        return self.config_result.temp_dirs.copy()


@contextmanager
def managed_environment(
    config_result: ConfigBuilderResult,
    extra_env: dict[str, str] | None = None,
) -> Generator[EnvironmentManager, None, None]:
    """Context manager function for environment management.

    Alternative function-based API for those who prefer functional style
    over class-based context managers.

    Args:
        config_result: Configuration result from config builders
        extra_env: Optional additional environment variables to apply

    Yields:
        EnvironmentManager instance

    Example:
        with managed_environment(get_qdrant_memory_config()) as env_mgr:
            # Test logic here
            assert env_mgr.get_env_var("VECTOR_DB_TYPE") == "qdrant"
    """
    with EnvironmentManager.from_config(config_result, extra_env) as manager:
        yield manager


@contextmanager
def quick_qdrant_env(
    shared_provider: Any = None,
    extra_env: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """Quick context manager for Qdrant memory environment setup.

    Simplified API for the most common use case where you just need
    the environment setup without needing to access the manager.

    Args:
        shared_provider: Optional shared sentence-transformers provider
        extra_env: Optional additional environment variables to apply

    Example:
        with quick_qdrant_env():
            # Test logic with Qdrant memory configuration
            result = subprocess.run([...])
            assert result.returncode == 0
    """
    with EnvironmentManager.qdrant_memory(shared_provider, extra_env):
        yield


@contextmanager
def quick_chroma_env(
    shared_provider: Any = None,
    extra_env: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """Quick context manager for ChromaDB memory environment setup.

    Simplified API for the most common use case where you just need
    the environment setup without needing to access the manager.

    Args:
        shared_provider: Optional shared sentence-transformers provider
        extra_env: Optional additional environment variables to apply

    Example:
        with quick_chroma_env():
            # Test logic with ChromaDB memory configuration
            result = subprocess.run([...])
            assert result.returncode == 0
    """
    with EnvironmentManager.chroma_memory(shared_provider, extra_env):
        yield


@contextmanager
def quick_text_only_env(
    extra_env: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """Quick context manager for text-only environment setup.

    Simplified API for text-only testing without vector database.

    Args:
        extra_env: Optional additional environment variables to apply

    Example:
        with quick_text_only_env():
            # Test logic with text-only configuration
            result = subprocess.run([...])
            assert result.returncode == 0
    """
    with EnvironmentManager.no_vector_db(extra_env):
        yield


# Migration helpers for existing test patterns
def migrate_try_finally_pattern(
    config_builder_func: callable,
    test_logic: callable,
    extra_env: dict[str, str] | None = None,
    **config_kwargs: Any,
) -> Any:
    """Helper function to migrate existing try/finally patterns.

    This function helps transition existing tests from manual try/finally
    blocks to the new context manager pattern.

    Args:
        config_builder_func: Configuration builder function (e.g., get_qdrant_memory_config)
        test_logic: Function containing the test logic
        extra_env: Optional additional environment variables
        **config_kwargs: Keyword arguments to pass to config builder

    Returns:
        Result of test_logic function

    Example:
        def test_something():
            # Old pattern:
            # config_result = get_qdrant_memory_config()
            # previous_env = apply_env_vars(config_result.env_vars)
            # try:
            #     # test logic
            # finally:
            #     restore_env_vars(previous_env)
            #     cleanup_temp_dirs(config_result.temp_dirs)

            # New pattern:
            return migrate_try_finally_pattern(
                get_qdrant_memory_config,
                lambda: subprocess.run([...])
            )
    """
    config_result = config_builder_func(**config_kwargs)
    with EnvironmentManager.from_config(config_result, extra_env):
        return test_logic()


# Common patterns for documentation and examples
class CommonPatterns:
    """Documentation class showing common usage patterns.

    This class serves as living documentation for the most frequently
    used environment management patterns in the test suite.
    """

    @staticmethod
    def basic_workflow_test() -> None:
        """Example: Basic workflow test pattern."""
        with EnvironmentManager.qdrant_memory():
            # Replace old pattern:
            # config_result = get_qdrant_memory_config()
            # previous_env = apply_env_vars(config_result.env_vars)
            # try:
            #     # Test logic
            # finally:
            #     restore_env_vars(previous_env)
            #     cleanup_temp_dirs(config_result.temp_dirs)

            # Test logic goes here
            pass

    @staticmethod
    def rag_generation_test() -> None:
        """Example: RAG generation test pattern."""
        with EnvironmentManager.rag_enabled():
            # Replace old pattern with custom env vars:
            # config_result = get_qdrant_memory_config()
            # env_vars = config_result.env_vars.copy()
            # env_vars.update({"GENERATION_ENABLE": "true", ...})
            # previous_env = apply_env_vars(env_vars)
            # try:
            #     # Test logic
            # finally:
            #     restore_env_vars(previous_env)
            #     cleanup_temp_dirs(config_result.temp_dirs)

            # RAG test logic goes here
            pass

    @staticmethod
    def configuration_matrix_test() -> None:
        """Example: Configuration matrix test pattern."""
        for provider in ["qdrant", "chroma"]:
            if provider == "qdrant":
                ctx = EnvironmentManager.qdrant_memory()
            else:
                ctx = EnvironmentManager.chroma_memory()

            with ctx:
                # Configuration-specific test logic
                pass

    @staticmethod
    def shared_provider_test(shared_model: Any) -> None:
        """Example: Shared provider optimization pattern."""
        with EnvironmentManager.qdrant_memory(shared_provider=shared_model):
            # Test logic with shared embedding model optimization
            pass
