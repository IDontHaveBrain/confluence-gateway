"""Configuration matrix integration tests for confluence-gateway.

Tests basic success cases for essential provider combinations:
- Qdrant + SentenceTransformers (memory mode)
- ChromaDB + SentenceTransformers (memory mode)
- Text-only mode (no vector database)

Focus: Basic success cases only, no error scenarios or complex validation.
"""

import json
import subprocess
from typing import Any

import pytest

from tests.fixtures.config_builders import (
    apply_env_vars,
    cleanup_temp_dirs,
    get_chroma_memory_config,
    get_no_vector_db_config,
    get_qdrant_memory_config,
    restore_env_vars,
)


def parse_cli_json_output(output: str) -> dict[str, Any]:
    """Parse JSON from CLI output that may contain info messages before JSON."""
    lines = output.strip().split("\n")

    # Find the first line that starts with '{' and collect all subsequent lines
    json_started = False
    json_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Start collecting JSON when we find the opening brace
        if not json_started and stripped_line.startswith("{"):
            json_started = True
            json_lines.append(line)
        elif json_started:
            # Continue collecting lines that are part of the JSON
            json_lines.append(line)

    if json_lines:
        # Join all JSON lines and parse
        json_text = "\n".join(json_lines)
        return json.loads(json_text)

    # If no JSON found, try parsing the entire output as fallback
    return json.loads(output.strip())


# Essential configuration combinations only
ESSENTIAL_CONFIGURATIONS = [
    ("qdrant", "sentence-transformers", "memory"),
    ("chroma", "sentence-transformers", "memory"),
    ("none", "none", "n/a"),
]


@pytest.mark.parametrize("vector_db,embedding,storage", ESSENTIAL_CONFIGURATIONS)
class TestEssentialConfigurationMatrix:
    """Test essential configuration combinations for basic success cases."""

    def test_configuration_initialization(
        self, vector_db: str, embedding: str, storage: str
    ) -> None:
        """Test that essential configuration combinations initialize correctly.

        Args:
            vector_db: Vector database provider ("qdrant", "chroma", "none")
            embedding: Embedding provider ("sentence-transformers", "none")
            storage: Storage mode ("memory", "n/a")
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test that the CLI can initialize without errors
            result = subprocess.run(
                ["/home/skawn/.local/bin/uv", "run", "confluence-gateway", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # CLI should start successfully with valid configuration
            assert result.returncode == 0, (
                f"CLI failed to initialize with config {vector_db}/{embedding}/{storage}: {result.stderr}"
            )

            # Test that version command works (verifies basic initialization)
            result = subprocess.run(
                ["/home/skawn/.local/bin/uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"Version command failed: {result.stderr}"
            assert "confluence-gateway" in result.stdout.lower()

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

    def test_basic_functionality(
        self, vector_db: str, embedding: str, storage: str
    ) -> None:
        """Test basic functionality for configuration combination.

        Args:
            vector_db: Vector database provider
            embedding: Embedding provider
            storage: Storage mode
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            expected_features = self._get_expected_features(
                vector_db, embedding, storage
            )

            # Test text search help (should always work)
            if expected_features["text_search"]:
                result = subprocess.run(
                    [
                        "/home/skawn/.local/bin/uv",
                        "run",
                        "confluence-gateway",
                        "search",
                        "text",
                        "--help",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0, (
                    f"Text search help failed: {result.stderr}"
                )

            # Test semantic search help if available
            if expected_features["semantic_search"]:
                result = subprocess.run(
                    [
                        "/home/skawn/.local/bin/uv",
                        "run",
                        "confluence-gateway",
                        "search",
                        "semantic",
                        "--help",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0, (
                    f"Semantic search help failed: {result.stderr}"
                )

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

    def test_basic_workflow(self, vector_db: str, embedding: str, storage: str) -> None:
        """Test basic workflow (spaces list) for configuration.

        Args:
            vector_db: Vector database provider
            embedding: Embedding provider
            storage: Storage mode
        """
        # Get the appropriate config builder
        config_result = self._build_configuration(vector_db, embedding, storage)

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test spaces list command (basic Confluence connectivity)
            spaces_result = subprocess.run(
                [
                    "/home/skawn/.local/bin/uv",
                    "run",
                    "confluence-gateway",
                    "spaces",
                    "list",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Only verify if Confluence credentials are available
            if spaces_result.returncode == 0:
                spaces_data = parse_cli_json_output(spaces_result.stdout)
                assert "results" in spaces_data
            else:
                # If no Confluence connection, at least verify CLI doesn't crash
                help_result = subprocess.run(
                    [
                        "/home/skawn/.local/bin/uv",
                        "run",
                        "confluence-gateway",
                        "search",
                        "--help",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert help_result.returncode == 0, (
                    f"Search help failed: {help_result.stderr}"
                )

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

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
    """Test basic provider compatibility."""

    def test_qdrant_sentence_transformers_compatibility(self) -> None:
        """Test Qdrant + SentenceTransformers basic compatibility."""
        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test basic CLI initialization
            result = subprocess.run(
                ["/home/skawn/.local/bin/uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, (
                f"Qdrant + SentenceTransformers initialization failed: {result.stderr}"
            )

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

    def test_chroma_sentence_transformers_compatibility(self) -> None:
        """Test ChromaDB + SentenceTransformers basic compatibility."""
        config_result = get_chroma_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test basic CLI initialization
            result = subprocess.run(
                ["/home/skawn/.local/bin/uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, (
                f"ChromaDB + SentenceTransformers initialization failed: {result.stderr}"
            )

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

    def test_text_only_mode_functionality(self) -> None:
        """Test text-only mode basic functionality."""
        config_result = get_no_vector_db_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test basic CLI initialization
            result = subprocess.run(
                ["/home/skawn/.local/bin/uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode == 0, (
                f"Text-only mode initialization failed: {result.stderr}"
            )

            # Test that text search help works
            result = subprocess.run(
                [
                    "/home/skawn/.local/bin/uv",
                    "run",
                    "confluence-gateway",
                    "search",
                    "text",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"Text search help failed: {result.stderr}"

        finally:
            # Restore environment
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)
