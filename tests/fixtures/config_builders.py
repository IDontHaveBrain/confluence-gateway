"""Simplified configuration factory for integration testing.

This module provides essential configuration builders for basic success testing:
- Memory-based Qdrant configuration
- Memory-based ChromaDB configuration
- Text-only configuration (no vector database)

Focuses on core functionality testing with minimal complexity.
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


def get_qdrant_memory_config() -> ConfigBuilderResult:
    """Get Qdrant memory configuration for basic testing."""
    temp_dirs = []

    # Build embedding config
    embedding_config = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        device="cpu",
    )

    # Build vector DB config
    vector_db_config = VectorDBConfig(
        type="qdrant",
        qdrant_url=":memory:",
        collection_name="test_confluence_embeddings",
        embedding_dimension=384,
    )

    # Environment variables
    env_vars = {
        "QDRANT_URL": ":memory:",
        "VECTOR_DB_TYPE": "qdrant",
        "VECTOR_DB_EMBEDDING_DIMENSION": "384",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSION": "384",
        "EMBEDDING_DEVICE": "cpu",
    }

    # Create user config file
    config_data = {
        "vector_db": {
            "type": "qdrant",
            "qdrant_url": ":memory:",
            "collection_name": "test_confluence_embeddings",
        },
        "embedding": {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "device": "cpu",
        },
    }

    config_file = _create_user_config_file(config_data)
    temp_dirs.append(config_file.parent)

    return ConfigBuilderResult(
        vector_db_config=vector_db_config,
        embedding_config=embedding_config,
        env_vars=env_vars,
        temp_dirs=temp_dirs,
        config_file_path=config_file,
    )


def get_chroma_memory_config() -> ConfigBuilderResult:
    """Get ChromaDB memory configuration for basic testing."""
    temp_dirs = []

    # Build embedding config
    embedding_config = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        device="cpu",
    )

    # Build vector DB config
    vector_db_config = VectorDBConfig(
        type="chroma",
        chroma_persist_path=None,  # Memory mode
        chroma_host=None,
        chroma_port=None,
        collection_name="test_confluence_embeddings",
        embedding_dimension=384,
    )

    # Environment variables
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

    # Create user config file
    config_data = {
        "vector_db": {
            "type": "chroma",
            "collection_name": "test_confluence_embeddings",
        },
        "embedding": {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "device": "cpu",
        },
    }

    config_file = _create_user_config_file(config_data)
    temp_dirs.append(config_file.parent)

    return ConfigBuilderResult(
        vector_db_config=vector_db_config,
        embedding_config=embedding_config,
        env_vars=env_vars,
        temp_dirs=temp_dirs,
        config_file_path=config_file,
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
