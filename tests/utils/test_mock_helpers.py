"""Tests for mock helpers utilities.

This module provides basic validation tests for the mock helpers to ensure
they work correctly and can be imported properly.
"""

import os
from unittest.mock import AsyncMock

import pytest

from tests.utils.mock_helpers import (
    ConfigurationMockHelper,
    EnvironmentContext,
    LiteLLMContext,
    MockContextFactory,
    MockMigrationHelper,
    PerformanceMockHelper,
    ResponseValidationHelper,
    SentenceTransformerContext,
)


class TestMockContextFactory:
    """Test the MockContextFactory class."""

    def test_factory_initialization(self):
        """Test that the factory can be initialized."""
        factory = MockContextFactory()
        assert factory is not None
        assert factory._active_contexts == []

    def test_litellm_context_creation(self):
        """Test LiteLLM context creation."""
        factory = MockContextFactory()
        with factory.create_litellm_context("test response") as mock_litellm:
            assert isinstance(mock_litellm, AsyncMock)
            assert mock_litellm._mock_name == "mock_litellm"

    def test_environment_context_creation(self):
        """Test environment context creation."""
        factory = MockContextFactory()
        test_vars = {"TEST_VAR": "test_value"}

        original_env = os.environ.copy()

        with factory.create_environment_context(test_vars) as env_vars:
            assert env_vars == test_vars
            assert os.environ.get("TEST_VAR") == "test_value"

        # Environment should be restored
        assert os.environ.copy() == original_env

    def test_sentence_transformer_context_with_mock(self):
        """Test sentence transformer context with mock model."""
        factory = MockContextFactory()

        with factory.create_sentence_transformer_context() as model:
            assert model is not None
            # Should have encode method
            assert hasattr(model, "encode")

    def test_full_test_context(self):
        """Test comprehensive test context creation."""
        factory = MockContextFactory()

        with factory.create_full_test_context(
            litellm_response="test response",
            env_vars={"TEST_KEY": "test_value"},
            mock_sentence_transformers=True,
        ) as context:
            assert "litellm_mock" in context
            assert "env_vars" in context
            assert "sentence_transformer_model" in context

            assert isinstance(context["litellm_mock"], AsyncMock)
            assert context["env_vars"]["TEST_KEY"] == "test_value"
            assert context["sentence_transformer_model"] is not None


class TestConvenienceContextManagers:
    """Test the convenience context managers."""

    def test_litellm_context(self):
        """Test LiteLLMContext convenience function."""
        with LiteLLMContext("test response") as mock_litellm:
            assert isinstance(mock_litellm, AsyncMock)

    def test_environment_context(self):
        """Test EnvironmentContext convenience function."""
        test_vars = {"TEST_VAR": "test_value"}
        original_env = os.environ.copy()

        with EnvironmentContext(test_vars) as env_vars:
            assert env_vars == test_vars
            assert os.environ.get("TEST_VAR") == "test_value"

        # Environment should be restored
        assert os.environ.copy() == original_env

    def test_sentence_transformer_context(self):
        """Test SentenceTransformerContext convenience function."""
        with SentenceTransformerContext() as model:
            assert model is not None


class TestConfigurationMockHelper:
    """Test the ConfigurationMockHelper class."""

    def test_temporary_config_path(self):
        """Test temporary config path creation."""
        with ConfigurationMockHelper.temporary_config_path() as config_path:
            assert config_path is not None
            assert isinstance(config_path, str)
            assert config_path.endswith(".json")

    def test_mock_confluence_credentials(self):
        """Test mock Confluence credentials."""
        with ConfigurationMockHelper.mock_confluence_credentials() as (
            url,
            user,
            token,
        ):
            assert url == "https://test.atlassian.net"
            assert user == "test@example.com"
            assert token == "test_api_token"

            # Environment should be set
            assert os.environ.get("CONFLUENCE_URL") == url
            assert os.environ.get("CONFLUENCE_USERNAME") == user
            assert os.environ.get("CONFLUENCE_API_TOKEN") == token


class TestResponseValidationHelper:
    """Test the ResponseValidationHelper class."""

    def test_validate_cli_json_response_success(self):
        """Test successful CLI JSON response validation."""
        response_data = {"results": [], "total": 0, "took_ms": 10}
        required_keys = ["results", "total", "took_ms"]

        # Should not raise any exception
        ResponseValidationHelper.validate_cli_json_response(
            response_data, required_keys, "test response"
        )

    def test_validate_cli_json_response_failure(self):
        """Test CLI JSON response validation failure."""
        response_data = {"results": []}
        required_keys = ["results", "total", "took_ms"]

        with pytest.raises(AssertionError) as exc_info:
            ResponseValidationHelper.validate_cli_json_response(
                response_data, required_keys, "test response"
            )

        assert "missing required key: total" in str(exc_info.value)


class TestPerformanceMockHelper:
    """Test the PerformanceMockHelper class."""

    def test_timed_mock_context(self):
        """Test timed mock context."""
        import time

        with PerformanceMockHelper.timed_mock_context("test_operation") as timer:
            time.sleep(0.01)  # Small delay for timing

        assert "duration" in timer
        assert timer["duration"] > 0
        assert timer["duration"] < 1.0  # Should be much less than 1 second

    def test_timed_mock_context_with_threshold(self):
        """Test timed mock context with performance threshold."""
        import time

        # Should pass with reasonable threshold
        with PerformanceMockHelper.timed_mock_context("fast_operation", 1.0) as timer:
            time.sleep(0.01)

        assert timer["duration"] < 1.0

    def test_timed_mock_context_threshold_exceeded(self):
        """Test timed mock context when threshold is exceeded."""
        import time

        # Should fail with very low threshold
        with pytest.raises(AssertionError) as exc_info:
            with PerformanceMockHelper.timed_mock_context("slow_operation", 0.001):
                time.sleep(0.01)

        assert "exceeded threshold" in str(exc_info.value)


class TestMockMigrationHelper:
    """Test the MockMigrationHelper class."""

    def test_migrate_litellm_pattern(self):
        """Test LiteLLM pattern migration helper."""
        context_manager = MockMigrationHelper.migrate_litellm_pattern("test content")

        with context_manager as mock_litellm:
            assert isinstance(mock_litellm, AsyncMock)

    def test_migrate_environment_pattern(self):
        """Test environment pattern migration helper."""
        env_vars = {"TEST_KEY": "test_value"}
        context_manager = MockMigrationHelper.migrate_environment_pattern(env_vars)

        with context_manager as applied_vars:
            assert applied_vars == env_vars
            assert os.environ.get("TEST_KEY") == "test_value"

    def test_show_migration_examples(self, capsys):
        """Test migration examples display."""
        MockMigrationHelper.show_migration_examples()

        captured = capsys.readouterr()
        assert "MOCK HELPERS MIGRATION EXAMPLES" in captured.out
        assert "LiteLLM Mocking:" in captured.out
        assert "Environment Management:" in captured.out


# Integration test to verify all imports work
def test_all_imports():
    """Test that all mock helpers can be imported successfully."""
    from tests.utils import (
        ConfigurationMockHelper,
        EnvironmentContext,
        LiteLLMContext,
        MockContextFactory,
        MockMigrationHelper,
        PerformanceMockHelper,
        ResponseValidationHelper,
        SentenceTransformerContext,
    )

    # All classes should be importable
    assert MockContextFactory is not None
    assert LiteLLMContext is not None
    assert EnvironmentContext is not None
    assert SentenceTransformerContext is not None
    assert ConfigurationMockHelper is not None
    assert ResponseValidationHelper is not None
    assert PerformanceMockHelper is not None
    assert MockMigrationHelper is not None
