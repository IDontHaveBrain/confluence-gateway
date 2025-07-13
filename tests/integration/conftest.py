"""Simplified integration test fixtures and configuration utilities.

This module provides essential pytest fixtures for basic success testing:
- Memory-mode vector database configurations (Qdrant, ChromaDB)
- Text-only configuration (no vector database)
- Environment variable injection and cleanup
- Memory mode enforcement for test isolation
"""

import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

import pytest
from confluence_gateway.core.config import (
    EmbeddingConfig,
    GenerationConfig,
    VectorDBConfig,
)


@pytest.fixture
def clean_environment() -> Generator[dict[str, str], None, None]:
    """Provide a clean environment for testing with original env restoration.

    Yields:
        Dictionary containing the original environment variables
    """
    # Save original environment
    original_env = os.environ.copy()

    # Clear vector DB and embedding related environment variables
    vector_db_vars = [
        "VECTOR_DB_TYPE",
        "VECTOR_DB_EMBEDDING_DIMENSION",
        "QDRANT_URL",
        "QDRANT_LOCAL_PATH",
        "QDRANT_API_KEY",
        "QDRANT_GRPC_PORT",
        "QDRANT_PREFER_GRPC",
        "CHROMA_PERSIST_PATH",
        "CHROMA_HOST",
        "CHROMA_PORT",
    ]

    embedding_vars = [
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_DEVICE",
        "LITELLM_API_KEY",
        "LITELLM_API_BASE",
    ]

    generation_vars = [
        "GENERATION_ENABLE",
        "GENERATION_PROVIDER",
        "GENERATION_MODEL_NAME",
        "GENERATION_LITELLM_API_KEY",
        "GENERATION_LITELLM_API_BASE",
        "GENERATION_PROMPT_TEMPLATE",
        "GENERATION_MAX_CONTEXT_TOKENS",
        "GENERATION_MAX_OUTPUT_TOKENS",
        "GENERATION_TEMPERATURE",
        "GENERATION_TIMEOUT",
    ]

    all_test_vars = vector_db_vars + embedding_vars + generation_vars

    for var in all_test_vars:
        if var in os.environ:
            del os.environ[var]

    yield original_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def config_injection():
    """Utility for injecting configuration via environment variables.

    Returns:
        Function to inject configuration into environment
    """

    def inject_config(config_dict: dict[str, Any]) -> None:
        """Inject configuration dictionary into environment variables.

        Args:
            config_dict: Configuration to inject as environment variables
        """
        for key, value in config_dict.items():
            if isinstance(value, bool):
                os.environ[key] = "true" if value else "false"
            elif isinstance(value, int | float):
                os.environ[key] = str(value)
            elif value is not None:
                os.environ[key] = str(value)

    return inject_config


@pytest.fixture
def qdrant_memory_config() -> dict[str, Any]:
    """Provide Qdrant memory-mode configuration.

    Returns:
        Configuration dictionary for memory-mode Qdrant
    """
    return {
        "VECTOR_DB_TYPE": "qdrant",
        "QDRANT_URL": ":memory:",
        "VECTOR_DB_EMBEDDING_DIMENSION": "384",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSION": "384",
        "EMBEDDING_DEVICE": "cpu",
    }


@pytest.fixture
def chroma_memory_config() -> dict[str, Any]:
    """Provide ChromaDB memory-mode configuration.

    Returns:
        Configuration dictionary for memory-mode ChromaDB
    """
    return {
        "VECTOR_DB_TYPE": "chroma",
        "CHROMA_PERSIST_PATH": "",  # Empty string for memory mode
        "VECTOR_DB_EMBEDDING_DIMENSION": "384",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSION": "384",
        "EMBEDDING_DEVICE": "cpu",
    }


@pytest.fixture
def no_vector_db_config() -> dict[str, Any]:
    """Provide text-only configuration (no vector database).

    Returns:
        Configuration dictionary for text-only mode
    """
    return {
        "VECTOR_DB_TYPE": "none",
        "EMBEDDING_PROVIDER": "none",
    }


@pytest.fixture
def config_builder():
    """Utility for building configuration objects from environment.

    Returns:
        Function to build configuration objects
    """

    def build_vector_db_config() -> VectorDBConfig:
        """Build VectorDBConfig from current environment variables."""
        return VectorDBConfig(
            type=os.environ.get("VECTOR_DB_TYPE", "none"),  # type: ignore[arg-type]
            embedding_dimension=int(
                os.environ.get("VECTOR_DB_EMBEDDING_DIMENSION", "384")
            ),
            qdrant_url=os.environ.get("QDRANT_URL"),  # type: ignore[arg-type]
            qdrant_local_path=os.environ.get("QDRANT_LOCAL_PATH"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            chroma_persist_path=os.environ.get("CHROMA_PERSIST_PATH"),
            chroma_host=os.environ.get("CHROMA_HOST"),
            chroma_port=int(os.environ["CHROMA_PORT"])
            if os.environ.get("CHROMA_PORT")
            else None,
        )

    def build_embedding_config() -> EmbeddingConfig:
        """Build EmbeddingConfig from current environment variables."""
        return EmbeddingConfig(
            provider=os.environ.get("EMBEDDING_PROVIDER", "none"),  # type: ignore[arg-type]
            model_name=os.environ.get("EMBEDDING_MODEL_NAME"),
            dimension=int(os.environ["EMBEDDING_DIMENSION"])
            if os.environ.get("EMBEDDING_DIMENSION")
            else None,
            litellm_api_key=os.environ.get("LITELLM_API_KEY"),
            litellm_api_base=os.environ.get("LITELLM_API_BASE"),  # type: ignore[arg-type]
            device=os.environ.get("EMBEDDING_DEVICE"),  # type: ignore[arg-type]
        )

    def build_generation_config() -> GenerationConfig:
        """Build GenerationConfig from current environment variables."""
        return GenerationConfig(
            enable=os.environ.get("GENERATION_ENABLE", "false").lower() == "true",
            provider="litellm",
            model_name=os.environ.get("GENERATION_MODEL_NAME"),
            litellm_api_key=os.environ.get("GENERATION_LITELLM_API_KEY"),
            litellm_api_base=os.environ.get("GENERATION_LITELLM_API_BASE"),  # type: ignore[arg-type]
            prompt_template=os.environ.get(
                "GENERATION_PROMPT_TEMPLATE",
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            ),
            max_context_tokens=int(
                os.environ.get("GENERATION_MAX_CONTEXT_TOKENS", "8000")
            ),
            max_output_tokens=int(
                os.environ.get("GENERATION_MAX_OUTPUT_TOKENS", "500")
            ),
            temperature=float(os.environ.get("GENERATION_TEMPERATURE", "0.1")),
            generation_timeout=int(os.environ.get("GENERATION_TIMEOUT", "60")),
        )

    return {
        "vector_db": build_vector_db_config,
        "embedding": build_embedding_config,
        "generation": build_generation_config,
    }


@pytest.fixture
def resource_cleanup():
    """Utility for cleaning up test resources.

    Returns:
        Function to clean up various resource types
    """
    cleanup_paths: list[Path] = []

    def register_path(path: Path) -> None:
        """Register a path for cleanup."""
        cleanup_paths.append(path)

    def cleanup_storage_paths() -> None:
        """Clean up all registered storage paths."""
        import shutil

        for path in cleanup_paths:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    yield {
        "register_path": register_path,
        "cleanup_storage": cleanup_storage_paths,
    }

    # Automatic cleanup on test completion
    import shutil

    for path in cleanup_paths:
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception:
                pass  # Best effort cleanup


@pytest.fixture(
    params=[
        "qdrant_memory",
        "chroma_memory",
        "no_vector_db",
    ]
)
def provider_config(
    request: pytest.FixtureRequest,
    qdrant_memory_config: dict[str, Any],
    chroma_memory_config: dict[str, Any],
    no_vector_db_config: dict[str, Any],
) -> dict[str, Any]:
    """Simplified parametrized fixture providing essential provider configurations.

    Args:
        request: pytest fixture request object
        qdrant_memory_config: Qdrant memory config
        chroma_memory_config: ChromaDB memory config
        no_vector_db_config: No vector DB config

    Returns:
        Configuration dictionary for the current parameter
    """
    config_map = {
        "qdrant_memory": qdrant_memory_config,
        "chroma_memory": chroma_memory_config,
        "no_vector_db": no_vector_db_config,
    }
    return config_map[request.param]


@pytest.fixture
def isolated_test_environment(
    clean_environment: dict[str, str],
    config_injection: Any,
    resource_cleanup: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Provide a completely isolated test environment.

    Args:
        clean_environment: Clean environment fixture
        config_injection: Configuration injection utility
        resource_cleanup: Resource cleanup utility

    Yields:
        Dictionary with utility functions for test isolation
    """
    yield {
        "inject_config": config_injection,
        "cleanup": resource_cleanup,
        "original_env": clean_environment,
    }

    # Ensure cleanup happens
    resource_cleanup["cleanup_storage"]()
