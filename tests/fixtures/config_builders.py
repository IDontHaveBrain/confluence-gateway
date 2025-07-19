"""Simplified configuration factory for integration testing.

This module provides essential configuration builders for basic success testing:
- Memory-based Qdrant configuration
- Memory-based ChromaDB configuration
- Text-only configuration (no vector database)
- Shared sentence-transformers provider injection for performance optimization

Focuses on core functionality testing with minimal complexity.

Shared Provider Support:
- Pass shared_provider parameter to avoid repeated model loading
- Uses USE_SHARED_EMBEDDING_PROVIDER environment flag
- Maintains backward compatibility with existing test patterns
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from confluence_gateway.core.config import (
    EmbeddingConfig,
    GenerationConfig,
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
    get_environment_context,
)


@dataclass
class ConfigBuilderResult:
    """Result container for configuration factory operations."""

    vector_db_config: VectorDBConfig | None = None
    embedding_config: EmbeddingConfig | None = None
    search_config: SearchConfig | None = None
    indexing_config: IndexingConfig | None = None
    generation_config: GenerationConfig | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    temp_dirs: list[Path] = field(default_factory=list)
    config_file_path: Path | None = None
    shared_provider: Any | None = (
        None  # Holds shared sentence-transformers provider instance
    )


def _create_temp_dir(prefix: str = "confluence_gateway_test_") -> Path:
    """Create a temporary directory for testing."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def _create_user_config_file(config_data: dict[str, Any]) -> Path:
    """Create a user configuration file."""
    temp_dir = _create_temp_dir("config_")
    config_file = temp_dir / "confluence_gateway_config.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    return config_file


def _get_shared_provider_env_vars(shared_provider: Any) -> dict[str, str]:
    """Get environment variables for shared provider usage."""
    if shared_provider is None:
        return {}

    return {
        "USE_SHARED_EMBEDDING_PROVIDER": "true",
        "SHARED_PROVIDER_MODEL_NAME": getattr(
            shared_provider, "_test_model_name", "all-MiniLM-L6-v2"
        ),
        "SHARED_PROVIDER_DEVICE": getattr(shared_provider, "_test_device", "cpu"),
    }


def _update_config_for_shared_provider(
    config_data: dict[str, Any], shared_provider: Any
) -> dict[str, Any]:
    """Update configuration data to use shared provider settings."""
    if shared_provider is None:
        return config_data

    # Add shared provider flag to embedding config
    if "embedding" in config_data:
        config_data["embedding"]["use_shared_provider"] = True
        config_data["embedding"]["shared_provider_id"] = id(shared_provider)

    return config_data


def create_shared_sentence_transformer_provider() -> Any:
    """Create a shared sentence-transformers provider for testing.

    Returns:
        Shared SentenceTransformer instance or None if not available
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Create shared provider with default test model
        provider = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Store model information as custom attributes for reference
        provider._test_model_name = "all-MiniLM-L6-v2"
        provider._test_device = "cpu"
        return provider
    except ImportError:
        # sentence-transformers not available, return None
        return None


def get_qdrant_memory_config(shared_provider: Any = None) -> ConfigBuilderResult:
    """Get Qdrant configuration for basic testing (memory or file based on testing mode).

    Args:
        shared_provider: Optional shared sentence-transformers provider to avoid repeated model loading

    Returns:
        ConfigBuilderResult with Qdrant configuration and optional shared provider
    """
    temp_dirs = []
    env_context = get_environment_context()
    use_memory = env_context.use_memory_mode

    # Build embedding config
    embedding_config = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        device="cpu",
    )

    # Build vector DB config based on testing mode
    if use_memory:
        vector_db_config = VectorDBConfig(
            type="qdrant",
            qdrant_url=":memory:",
            collection_name="test_cg_embeddings",
            embedding_dimension=384,
        )
        env_vars = {
            "QDRANT_URL": ":memory:",
            "VECTOR_DB_TYPE": "qdrant",
            "VECTOR_DB_EMBEDDING_DIMENSION": "384",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "EMBEDDING_DEVICE": "cpu",
        }
        config_data = {
            "vector_db": {
                "type": "qdrant",
                "qdrant_url": ":memory:",
                "collection_name": "test_cg_embeddings",
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu",
            },
        }
    else:
        # CI mode - use file storage (path will be set by conftest.py)
        qdrant_path = os.environ.get("QDRANT_LOCAL_PATH", "")
        vector_db_config = VectorDBConfig(
            type="qdrant",
            qdrant_local_path=qdrant_path,
            collection_name="test_cg_embeddings",
            embedding_dimension=384,
        )
        env_vars = {
            "QDRANT_LOCAL_PATH": qdrant_path,
            "VECTOR_DB_TYPE": "qdrant",
            "VECTOR_DB_EMBEDDING_DIMENSION": "384",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "EMBEDDING_DEVICE": "cpu",
        }
        config_data = {
            "vector_db": {
                "type": "qdrant",
                "qdrant_local_path": qdrant_path,
                "collection_name": "test_cg_embeddings",
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu",
            },
        }

    # Add shared provider environment variables if provided
    if shared_provider is not None:
        env_vars.update(_get_shared_provider_env_vars(shared_provider))

    # Update config for shared provider if provided
    config_data = _update_config_for_shared_provider(config_data, shared_provider)

    config_file = _create_user_config_file(config_data)
    temp_dirs.append(config_file.parent)

    return ConfigBuilderResult(
        vector_db_config=vector_db_config,
        embedding_config=embedding_config,
        env_vars=env_vars,
        temp_dirs=temp_dirs,
        config_file_path=config_file,
        shared_provider=shared_provider,
    )


def get_chroma_memory_config(shared_provider: Any = None) -> ConfigBuilderResult:
    """Get ChromaDB configuration for basic testing (memory or file based on testing mode).

    Args:
        shared_provider: Optional shared sentence-transformers provider to avoid repeated model loading

    Returns:
        ConfigBuilderResult with ChromaDB configuration and optional shared provider
    """
    temp_dirs = []
    env_context = get_environment_context()
    use_memory = env_context.use_memory_mode

    # Build embedding config
    embedding_config = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        device="cpu",
    )

    # Build vector DB config based on testing mode
    if use_memory:
        vector_db_config = VectorDBConfig(
            type="chroma",
            chroma_persist_path=None,  # Memory mode
            chroma_host=None,
            chroma_port=None,
            collection_name="test_cg_embeddings",
            embedding_dimension=384,
        )
        env_vars = {
            "CHROMA_PERSIST_PATH": "",
            "CHROMA_HOST": "",
            "CHROMA_PORT": "",
            "VECTOR_DB_TYPE": "chroma",
            "VECTOR_DB_EMBEDDING_DIMENSION": "384",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "EMBEDDING_DEVICE": "cpu",
        }
        config_data = {
            "vector_db": {
                "type": "chroma",
                "collection_name": "test_cg_embeddings",
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu",
            },
        }
    else:
        # CI mode - use file storage (path will be set by conftest.py)
        chroma_path = os.environ.get("CHROMA_PERSIST_PATH", "")
        vector_db_config = VectorDBConfig(
            type="chroma",
            chroma_persist_path=chroma_path,
            chroma_host=None,
            chroma_port=None,
            collection_name="test_cg_embeddings",
            embedding_dimension=384,
        )
        env_vars = {
            "CHROMA_PERSIST_PATH": chroma_path,
            "CHROMA_HOST": "",
            "CHROMA_PORT": "",
            "VECTOR_DB_TYPE": "chroma",
            "VECTOR_DB_EMBEDDING_DIMENSION": "384",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "EMBEDDING_DEVICE": "cpu",
        }
        config_data = {
            "vector_db": {
                "type": "chroma",
                "chroma_persist_path": chroma_path,
                "collection_name": "test_cg_embeddings",
            },
            "embedding": {
                "provider": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu",
            },
        }

    # Add shared provider environment variables if provided
    if shared_provider is not None:
        env_vars.update(_get_shared_provider_env_vars(shared_provider))

    # Update config for shared provider if provided
    config_data = _update_config_for_shared_provider(config_data, shared_provider)

    config_file = _create_user_config_file(config_data)
    temp_dirs.append(config_file.parent)

    return ConfigBuilderResult(
        vector_db_config=vector_db_config,
        embedding_config=embedding_config,
        env_vars=env_vars,
        temp_dirs=temp_dirs,
        config_file_path=config_file,
        shared_provider=shared_provider,
    )


def get_no_vector_db_config() -> ConfigBuilderResult:
    """Get text-only configuration for basic testing."""
    temp_dirs = []

    # Build configs
    search_config = SearchConfig(hybrid_search_enabled=False)
    indexing_config = IndexingConfig()

    # Environment variables
    env_vars = {
        "VECTOR_DB_TYPE": "none",
        "EMBEDDING_PROVIDER": "none",
    }

    # Create user config file
    config_data = {
        "vector_db": {"type": "none"},
        "embedding": {"provider": "none"},
        "search": {"hybrid_search_enabled": False},
    }

    config_file = _create_user_config_file(config_data)
    temp_dirs.append(config_file.parent)

    return ConfigBuilderResult(
        vector_db_config=None,
        embedding_config=None,
        search_config=search_config,
        indexing_config=indexing_config,
        env_vars=env_vars,
        temp_dirs=temp_dirs,
        config_file_path=config_file,
    )


def apply_env_vars(env_vars: dict[str, str]) -> dict[str, str | None]:
    """Apply environment variables and return previous values."""
    previous_values = {}
    for key, value in env_vars.items():
        previous_values[key] = os.environ.get(key)
        os.environ[key] = value
    return previous_values


def restore_env_vars(previous_values: dict[str, str | None]) -> None:
    """Restore environment variables to previous values."""
    for key, value in previous_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def cleanup_temp_dirs(temp_dirs: list[Path]) -> None:
    """Clean up temporary directories."""
    import shutil

    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# Convenience functions for shared provider management


def get_qdrant_with_shared_provider() -> tuple[ConfigBuilderResult, Any]:
    """Get Qdrant config with a shared sentence-transformers provider (memory or file based on testing mode).

    Returns:
        Tuple of (ConfigBuilderResult, shared_provider) for convenience
    """
    shared_provider = create_shared_sentence_transformer_provider()
    config_result = get_qdrant_memory_config(shared_provider=shared_provider)
    return config_result, shared_provider


def get_chroma_with_shared_provider() -> tuple[ConfigBuilderResult, Any]:
    """Get ChromaDB config with a shared sentence-transformers provider (memory or file based on testing mode).

    Returns:
        Tuple of (ConfigBuilderResult, shared_provider) for convenience
    """
    shared_provider = create_shared_sentence_transformer_provider()
    config_result = get_chroma_memory_config(shared_provider=shared_provider)
    return config_result, shared_provider


def get_multi_provider_shared_configs(
    use_shared_provider: bool = True,
) -> tuple[list[ConfigBuilderResult], Any | None]:
    """Get multiple provider configurations with optional shared sentence-transformers provider.

    Args:
        use_shared_provider: Whether to use a shared provider across all configs

    Returns:
        Tuple of (list of ConfigBuilderResult, shared_provider or None)

    Example:
        configs, shared_provider = get_multi_provider_shared_configs(use_shared_provider=True)
        qdrant_config, chroma_config = configs
    """
    shared_provider = None
    if use_shared_provider:
        shared_provider = create_shared_sentence_transformer_provider()

    configs = [
        get_qdrant_memory_config(shared_provider=shared_provider),
        get_chroma_memory_config(shared_provider=shared_provider),
    ]

    return configs, shared_provider
