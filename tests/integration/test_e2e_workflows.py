"""End-to-end workflow integration tests for Confluence Gateway.

This module provides basic workflow tests that validate essential user scenarios
with focus on success case validation only.
"""

import json
import os
import subprocess
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.config_builders import (
    apply_env_vars,
    cleanup_temp_dirs,
    get_qdrant_memory_config,
    restore_env_vars,
)


def parse_cli_json_output(output: str) -> dict[str, Any]:
    """Parse JSON from CLI output that may contain info messages before JSON.

    Args:
        output: Raw CLI output string

    Returns:
        Parsed JSON data as dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON found in output
    """
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


def create_mock_litellm_response(content: str) -> MagicMock:
    """Create a mock LiteLLM response object.

    Args:
        content: Response content to return

    Returns:
        Mock response object matching LiteLLM structure
    """
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    return mock_response


def get_confluence_credentials() -> tuple[str, str, str]:
    """Get Confluence credentials from environment variables.

    Returns:
        Tuple of (url, username, api_token)

    Raises:
        pytest.skip: If required credentials are not available
    """
    url = os.getenv("CONFLUENCE_URL")
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not all([url, username, api_token]):
        pytest.skip("Confluence credentials not available in environment")

    return url, username, api_token


class TestBasicIndexingWorkflow:
    """Test basic indexing workflow success cases only."""

    def test_basic_indexing_pipeline(self) -> None:
        """Test basic indexing workflow from space listing to content storage."""
        get_confluence_credentials()

        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Step 1: List available spaces
            spaces_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "spaces",
                    "list",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert spaces_result.returncode == 0
            spaces_data = parse_cli_json_output(spaces_result.stdout)
            assert "spaces" in spaces_data
            assert len(spaces_data["spaces"]) > 0

            # Step 2: Index first available space
            test_space_key = spaces_data["spaces"][0]["key"]
            index_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "index",
                    "trigger",
                    "--space",
                    test_space_key,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Trigger command only returns success/failure status, not JSON
            assert index_result.returncode == 0

            # Check indexing status via status command to get JSON data
            status_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "index",
                    "status",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert status_result.returncode == 0
            status_data = parse_cli_json_output(status_result.stdout)
            assert "status" in status_data

        finally:
            # Restore environment and cleanup
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)


class TestBasicTextSearchWorkflow:
    """Test basic text search functionality."""

    def test_basic_text_search(self) -> None:
        """Test basic text search functionality."""
        get_confluence_credentials()

        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            search_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "search",
                    "text",
                    "documentation",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert search_result.returncode == 0
            search_data = parse_cli_json_output(search_result.stdout)

            # Validate search response structure
            assert "results" in search_data
            assert "total" in search_data
            assert "took_ms" in search_data

        finally:
            # Restore environment and cleanup
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)


class TestBasicSemanticSearchWorkflow:
    """Test basic semantic search functionality."""

    def test_basic_semantic_search(self) -> None:
        """Test basic semantic search functionality."""
        get_confluence_credentials()

        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # First index some content
            spaces_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "spaces",
                    "list",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert spaces_result.returncode == 0
            spaces_data = parse_cli_json_output(spaces_result.stdout)
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "index",
                    "trigger",
                    "--space",
                    test_space_key,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert index_result.returncode == 0

            # Perform semantic search
            search_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "search",
                    "semantic",
                    "software development",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert search_result.returncode == 0
            search_data = parse_cli_json_output(search_result.stdout)

            # Validate semantic search response
            assert "results" in search_data
            assert "query" in search_data
            assert "count" in search_data

        finally:
            # Restore environment and cleanup
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)


class TestBasicHybridSearchWorkflow:
    """Test basic hybrid search functionality."""

    def test_basic_hybrid_search(self) -> None:
        """Test basic hybrid search functionality."""
        get_confluence_credentials()

        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # First index some content
            spaces_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "spaces",
                    "list",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert spaces_result.returncode == 0
            spaces_data = parse_cli_json_output(spaces_result.stdout)
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "index",
                    "trigger",
                    "--space",
                    test_space_key,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert index_result.returncode == 0

            # Perform hybrid search
            search_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "search",
                    "text",
                    "development process",
                    "--hybrid",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert search_result.returncode == 0
            search_data = parse_cli_json_output(search_result.stdout)

            # Validate hybrid search response
            assert "results" in search_data

        finally:
            # Restore environment and cleanup
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)


class TestBasicRAGWorkflow:
    """Test basic RAG workflow functionality."""

    def test_basic_rag_generation(self) -> None:
        """Test basic RAG answer generation workflow."""
        get_confluence_credentials()

        config_result = get_qdrant_memory_config()

        # Add generation environment variables
        env_vars = config_result.env_vars.copy()
        env_vars.update(
            {
                "GENERATION_ENABLE": "true",
                "GENERATION_MODEL_NAME": "openrouter/google/gemini-2.5-flash",
                "GENERATION_LITELLM_API_KEY": "test_api_key",
            }
        )

        # Apply environment variables
        previous_env = apply_env_vars(env_vars)

        try:
            # First index some content
            spaces_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "spaces",
                    "list",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert spaces_result.returncode == 0
            spaces_data = parse_cli_json_output(spaces_result.stdout)
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "confluence-gateway",
                    "index",
                    "trigger",
                    "--space",
                    test_space_key,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert index_result.returncode == 0

            # Generate answer using RAG with mocked LiteLLM
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
                mock_litellm.return_value = create_mock_litellm_response(
                    "Based on the retrieved documentation, software development best practices "
                    "include following coding standards, implementing proper testing strategies, "
                    "using version control, and maintaining clear documentation."
                )

                generate_result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "confluence-gateway",
                        "generate",
                        "answer",
                        "What are the software development best practices?",
                    ],
                    env={**os.environ, **env_vars},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                assert generate_result.returncode == 0
                generate_data = parse_cli_json_output(generate_result.stdout)

                # Validate RAG response structure
                assert "answer" in generate_data
                assert "sources" in generate_data

        finally:
            # Restore environment and cleanup
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)
