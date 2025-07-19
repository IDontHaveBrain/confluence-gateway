"""End-to-end workflow integration tests for Confluence Gateway.

This module provides basic workflow tests that validate essential user scenarios
with focus on success case validation only.
"""

import os
from typing import Any

import pytest

from tests.utils.cli_helpers import (
    CLITestRunner,
    run_generate_answer,
    run_search_hybrid,
    run_search_semantic,
    run_search_text,
    run_spaces_list,
)
from tests.utils.environment_helpers import EnvironmentManager
from tests.utils.mock_helpers import LiteLLMContext


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

        with EnvironmentManager.qdrant_memory():
            # Step 1: List available spaces
            spaces_data = run_spaces_list()
            assert "spaces" in spaces_data
            assert len(spaces_data["spaces"]) > 0

            # Step 2: Index first available space
            test_space_key = spaces_data["spaces"][0]["key"]
            index_result = CLITestRunner.run_command(
                ["index", "trigger", "--space", test_space_key], timeout=60
            )
            # Trigger command only returns success/failure status, not JSON
            assert index_result.returncode == 0

            # Check indexing status via status command to get JSON data
            status_data = CLITestRunner.run_command_json(["index", "status"])
            assert "status" in status_data


class TestBasicTextSearchWorkflow:
    """Test basic text search functionality."""

    def test_basic_text_search(self) -> None:
        """Test basic text search functionality."""
        get_confluence_credentials()

        with EnvironmentManager.qdrant_memory():
            search_data = run_search_text("documentation")

            # Validate search response structure
            CLITestRunner.assert_search_response(
                search_data, expected_fields=["total", "took_ms"]
            )


class TestBasicSemanticSearchWorkflow:
    """Test basic semantic search functionality."""

    def test_basic_semantic_search(self) -> None:
        """Test basic semantic search functionality."""
        get_confluence_credentials()

        with EnvironmentManager.qdrant_memory():
            # First index some content
            spaces_data = run_spaces_list()
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = CLITestRunner.run_command(
                ["index", "trigger", "--space", test_space_key], timeout=60
            )
            assert index_result.returncode == 0

            # Perform semantic search
            search_data = run_search_semantic("software development")

            # Validate semantic search response
            CLITestRunner.assert_search_response(
                search_data, expected_fields=["query", "count"]
            )


class TestBasicHybridSearchWorkflow:
    """Test basic hybrid search functionality."""

    def test_basic_hybrid_search(self) -> None:
        """Test basic hybrid search functionality."""
        get_confluence_credentials()

        with EnvironmentManager.qdrant_memory():
            # First index some content
            spaces_data = run_spaces_list()
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = CLITestRunner.run_command(
                ["index", "trigger", "--space", test_space_key], timeout=60
            )
            assert index_result.returncode == 0

            # Perform hybrid search
            # Increased timeout to 30s to accommodate model initialization in various CI environments
            search_data = run_search_hybrid("development process", timeout=30)

            # Validate hybrid search response
            CLITestRunner.assert_search_response(search_data)


class TestBasicRAGWorkflow:
    """Test basic RAG workflow functionality."""

    def test_basic_rag_generation(self) -> None:
        """Test basic RAG answer generation workflow."""
        get_confluence_credentials()

        with EnvironmentManager.rag_enabled():
            # First index some content
            spaces_data = run_spaces_list()
            test_space_key = spaces_data["spaces"][0]["key"]

            # Index the space
            index_result = CLITestRunner.run_command(
                ["index", "trigger", "--space", test_space_key], timeout=60
            )
            assert index_result.returncode == 0

            # Generate answer using RAG with mocked LiteLLM
            with LiteLLMContext(
                "Based on the retrieved documentation, software development best practices "
                "include following coding standards, implementing proper testing strategies, "
                "using version control, and maintaining clear documentation."
            ):
                generate_data = run_generate_answer(
                    "What are the software development best practices?", timeout=30
                )

                # Validate RAG response structure
                CLITestRunner.assert_generation_response(generate_data)
                assert "sources" in generate_data
