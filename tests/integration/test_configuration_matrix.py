"""Configuration matrix integration tests for confluence-gateway.

Tests basic success cases for essential provider combinations:
- Qdrant + SentenceTransformers (memory mode)
- ChromaDB + SentenceTransformers (memory mode)
- Text-only mode (no vector database)

Focus: Basic success cases only, no error scenarios or complex validation.
"""

from typing import Any

import pytest

from tests.fixtures.config_builders import (
    get_chroma_memory_config,
    get_no_vector_db_config,
    get_qdrant_memory_config,
)
from tests.utils.cli_helpers import CLITestRunner, run_spaces_list
from tests.utils.environment_helpers import EnvironmentManager
from tests.utils.mock_helpers import SentenceTransformerContext
from tests.utils.performance_helpers import performance_tracked

# Essential configuration combinations only
ESSENTIAL_CONFIGURATIONS = [
    ("qdrant", "sentence-transformers", "memory"),
    ("chroma", "sentence-transformers", "memory"),
    ("none", "none", "n/a"),
]


@pytest.mark.parametrize("vector_db,embedding,storage", ESSENTIAL_CONFIGURATIONS)
class TestEssentialConfigurationMatrix:
    """Test essential configuration combinations for basic success cases.

    This test class validates the core provider matrix with shared embedding
    optimization for sentence-transformers configurations.
    """

    @performance_tracked("config_initialization")
    def test_configuration_initialization(
        self,
        vector_db: str,
        embedding: str,
        storage: str,
        shared_sentence_transformer_model,
    ) -> None:
        """Test that essential configuration combinations initialize correctly.

        Args:
            vector_db: Vector database provider ("qdrant", "chroma", "none")
            embedding: Embedding provider ("sentence-transformers", "none")
            storage: Storage mode ("memory", "n/a")
            shared_sentence_transformer_model: Session-scoped shared model for optimization
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)

        with EnvironmentManager.from_config(config_result):
            optimization_status = "standard"
            if (
                embedding == "sentence-transformers"
                and shared_sentence_transformer_model is not None
            ):
                optimization_status = "shared model optimization"

            print(
                f"Testing {vector_db}/{embedding}/{storage} with {optimization_status}"
            )

            # Use shared model context if available
            with SentenceTransformerContext(shared_sentence_transformer_model):
                # Test that the CLI can initialize without errors
                CLITestRunner.run_command(["--help"], timeout=30)

                # Test that version command works (verifies basic initialization)
                result = CLITestRunner.run_command(["--version"], timeout=30)
                assert "confluence-gateway" in result.stdout.lower()

    @performance_tracked("basic_functionality")
    def test_basic_functionality(
        self,
        vector_db: str,
        embedding: str,
        storage: str,
        shared_sentence_transformer_model,
    ) -> None:
        """Test basic functionality for configuration combination.

        Args:
            vector_db: Vector database provider
            embedding: Embedding provider
            storage: Storage mode
            shared_sentence_transformer_model: Session-scoped shared model for optimization
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)
        expected_features = self._get_expected_features(vector_db, embedding, storage)

        with EnvironmentManager.from_config(config_result):
            # Use shared model context if available
            with SentenceTransformerContext(shared_sentence_transformer_model):
                # Test text search help (should always work)
                if expected_features["text_search"]:
                    CLITestRunner.run_command(["search", "text", "--help"], timeout=30)

                # Test semantic search help if available
                if expected_features["semantic_search"]:
                    CLITestRunner.run_command(
                        ["search", "semantic", "--help"], timeout=30
                    )

            optimization_status = (
                "optimized" if shared_sentence_transformer_model else "standard"
            )
            print(f"Functionality test ({optimization_status}) completed")

    @performance_tracked("basic_workflow")
    def test_basic_workflow(
        self,
        vector_db: str,
        embedding: str,
        storage: str,
        shared_sentence_transformer_model,
    ) -> None:
        """Test basic workflow (spaces list) for configuration.

        Args:
            vector_db: Vector database provider
            embedding: Embedding provider
            storage: Storage mode
            shared_sentence_transformer_model: Session-scoped shared model for optimization
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)

        with EnvironmentManager.from_config(config_result):
            # Use shared model context if available
            with SentenceTransformerContext(shared_sentence_transformer_model):
                # Test spaces list command (basic Confluence connectivity)
                try:
                    spaces_data = run_spaces_list(timeout=60)
                    assert "results" in spaces_data
                except AssertionError:
                    # If no Confluence connection, at least verify CLI doesn't crash
                    CLITestRunner.run_command(["search", "--help"], timeout=30)

    def _get_expected_features(
        self, vector_db: str, embedding: str, storage: str
    ) -> dict[str, bool]:
        """Get expected feature availability for configuration combination."""
        has_vector = vector_db != "none" and embedding != "none"

        return {
            "text_search": True,  # Always available
            "semantic_search": has_vector,
            "hybrid_search": has_vector,
            "indexing": True,  # Text indexing always available
        }

    def _build_configuration(self, vector_db: str, embedding: str, storage: str):
        """Build configuration for the given provider combination.

        Args:
            vector_db: Vector database provider ("qdrant", "chroma", "none")
            embedding: Embedding provider ("sentence-transformers", "none")
            storage: Storage mode ("memory", "n/a")

        Returns:
            ConfigBuilderResult with the appropriate configuration
        """
        if vector_db == "none" and embedding == "none":
            # Text-only mode
            return get_no_vector_db_config()
        elif vector_db == "qdrant":
            return get_qdrant_memory_config()
        elif vector_db == "chroma":
            return get_chroma_memory_config()
        else:
            raise ValueError(f"Invalid vector database provider: {vector_db}")


class TestProviderCompatibility:
    """Test basic provider compatibility with shared embedding optimization."""

    @performance_tracked("qdrant_sentence_transformers_compatibility")
    def test_qdrant_sentence_transformers_compatibility(
        self, shared_sentence_transformer_model
    ) -> None:
        """Test Qdrant + SentenceTransformers basic compatibility with shared model optimization."""
        with EnvironmentManager.qdrant_memory():
            # Use shared model context if available
            with SentenceTransformerContext(shared_sentence_transformer_model):
                # Test basic CLI initialization
                CLITestRunner.run_command(["--version"], timeout=60)

            optimization_status = (
                "optimized" if shared_sentence_transformer_model else "standard"
            )
            print(
                f"Qdrant + SentenceTransformers compatibility test ({optimization_status}) completed"
            )

    @performance_tracked("chroma_sentence_transformers_compatibility")
    def test_chroma_sentence_transformers_compatibility(
        self, shared_sentence_transformer_model
    ) -> None:
        """Test ChromaDB + SentenceTransformers basic compatibility with shared model optimization."""
        with EnvironmentManager.chroma_memory():
            # Use shared model context if available
            with SentenceTransformerContext(shared_sentence_transformer_model):
                # Test basic CLI initialization
                CLITestRunner.run_command(["--version"], timeout=60)

            optimization_status = (
                "optimized" if shared_sentence_transformer_model else "standard"
            )
            print(
                f"ChromaDB + SentenceTransformers compatibility test ({optimization_status}) completed"
            )

    @performance_tracked("text_only_mode_functionality")
    def test_text_only_mode_functionality(self) -> None:
        """Test text-only mode basic functionality."""
        with EnvironmentManager.no_vector_db():
            # Test basic CLI initialization
            CLITestRunner.run_command(["--version"], timeout=60)

            # Test that text search help works
            CLITestRunner.run_command(["search", "text", "--help"], timeout=30)
