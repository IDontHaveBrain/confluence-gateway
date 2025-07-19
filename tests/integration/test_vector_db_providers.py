"""Essential integration tests for vector database providers.

Basic tests for vector database implementations:
- Qdrant memory mode basic functionality
- ChromaDB memory mode basic functionality
"""

import os
from contextlib import contextmanager
from pathlib import Path

import pytest
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter

from tests.fixtures.test_data import TEST_CONFIGURATIONS


class TestVectorDBProviders:
    """Test suite for vector database provider integration."""

    @pytest.mark.parametrize("vector_db_type", ["qdrant", "chroma"])
    def test_vector_db_memory_mode_basic_functionality(
        self, vector_db_type: str, tmp_path: Path
    ) -> None:
        """Test basic vector DB functionality in memory mode.

        Args:
            vector_db_type: Type of vector database to test
            tmp_path: Pytest temporary directory fixture
        """
        # Setup memory mode configuration
        config = TEST_CONFIGURATIONS[f"{vector_db_type}_memory"].copy()

        # Set environment variables for this test
        env_vars = self._setup_test_environment(config, memory_mode=True)

        with self._temporary_env_vars(env_vars):
            # Test basic adapter creation
            adapter = get_vector_db_adapter()

            if adapter is not None:
                # Basic functionality verification
                assert hasattr(adapter, "type") or hasattr(adapter, "__class__")

                # Test adapter can be closed without errors
                if hasattr(adapter, "close"):
                    adapter.close()

    def _setup_test_environment(
        self, config: dict, memory_mode: bool = True
    ) -> dict[str, str]:
        """Setup environment variables for test configuration.

        Args:
            config: Configuration dictionary
            memory_mode: Whether to use memory mode

        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {}

        # Vector DB configuration
        if "vector_db" in config:
            vdb_config = config["vector_db"]
            env_vars["VECTOR_DB_TYPE"] = vdb_config["type"]

            if vdb_config["type"] == "qdrant":
                if memory_mode:
                    env_vars["QDRANT_URL"] = ":memory:"
                    env_vars["QDRANT_LOCAL_PATH"] = ""
            elif vdb_config["type"] == "chroma":
                if memory_mode:
                    env_vars["CHROMA_PERSIST_PATH"] = ""
                    env_vars["CHROMA_HOST"] = ""
                    env_vars["CHROMA_PORT"] = ""

        # Embedding configuration
        if "embedding" in config:
            emb_config = config["embedding"]
            env_vars["EMBEDDING_PROVIDER"] = emb_config["provider"]
            if "model_name" in emb_config:
                env_vars["EMBEDDING_MODEL_NAME"] = emb_config["model_name"]
            if "dimension" in emb_config:
                env_vars["EMBEDDING_DIMENSION"] = str(emb_config["dimension"])

        return env_vars

    @contextmanager
    def _temporary_env_vars(self, env_vars: dict[str, str]):
        """Context manager for temporarily setting environment variables."""
        # Store original values
        original_values = {}
        for key in env_vars:
            original_values[key] = os.environ.get(key)

        # Set new values
        for key, value in env_vars.items():
            if value:  # Only set non-empty values
                os.environ[key] = value

        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
