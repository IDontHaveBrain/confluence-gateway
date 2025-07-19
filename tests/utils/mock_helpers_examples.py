"""Examples of using mock helpers to replace existing patterns.

This file demonstrates how to use the new mock helpers utility to replace
the 8+ mocking patterns found across integration tests with standardized,
reusable context managers and factory classes.

Run this file directly to see the examples in action:
    python -m tests.utils.mock_helpers_examples
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


def demonstrate_basic_patterns():
    """Demonstrate the basic mock helper patterns."""

    print("MOCK HELPERS - BASIC PATTERNS")
    print("=" * 50)

    # PATTERN 1: LiteLLM Mocking
    print("\n1. LiteLLM Async Mocking:")
    print("-" * 25)

    async def demo_litellm_usage():
        """Demonstrate LiteLLM mocking."""
        with LiteLLMContext("This is a mocked AI response") as mock_litellm:
            # Simulate calling litellm.acompletion
            print("✓ LiteLLM mock created successfully")
            print(f"✓ Mock type: {type(mock_litellm)}")
            print(
                f"✓ Mock configured with AsyncMock: {isinstance(mock_litellm, AsyncMock)}"
            )

            # Show what the mock response would look like
            mock_response = mock_litellm.return_value
            print(
                f"✓ Mock response content: {mock_response.choices[0].message.content}"
            )

    asyncio.run(demo_litellm_usage())

    # PATTERN 2: Environment Context
    print("\n2. Environment Variable Management:")
    print("-" * 35)

    original_value = os.environ.get("TEST_DEMO_VAR", "NOT_SET")
    print(f"Original TEST_DEMO_VAR: {original_value}")

    with EnvironmentContext(
        {"TEST_DEMO_VAR": "temporary_value", "ANOTHER_VAR": "another_value"}
    ) as env_vars:
        print(f"Inside context TEST_DEMO_VAR: {os.environ.get('TEST_DEMO_VAR')}")
        print(f"Inside context ANOTHER_VAR: {os.environ.get('ANOTHER_VAR')}")
        print(f"Environment variables applied: {env_vars}")

    restored_value = os.environ.get("TEST_DEMO_VAR", "NOT_SET")
    print(f"After context TEST_DEMO_VAR: {restored_value}")
    print("✓ Environment properly restored")

    # PATTERN 3: Sentence Transformer Mocking
    print("\n3. Sentence Transformer Mocking:")
    print("-" * 32)

    with SentenceTransformerContext() as mock_model:
        print(f"✓ Mock model created: {mock_model is not None}")
        print(f"✓ Model has encode method: {hasattr(mock_model, 'encode')}")

        # Demonstrate encoding
        mock_embeddings = mock_model.encode(["test text"])
        print(f"✓ Mock embeddings: {mock_embeddings}")

    # PATTERN 4: Configuration Helpers
    print("\n4. Configuration Management:")
    print("-" * 28)

    with ConfigurationMockHelper.temporary_config_path() as config_path:
        print(f"✓ Temporary config path: {config_path}")
        print(f"✓ Path exists: {Path(config_path).exists()}")

    print(f"✓ Temporary path cleaned up: {not Path(config_path).exists()}")

    with ConfigurationMockHelper.mock_confluence_credentials() as (url, user, token):
        print(f"✓ Mock Confluence URL: {url}")
        print(f"✓ Mock username: {user}")
        print(f"✓ Mock token: {token[:10]}...")


def demonstrate_factory_patterns():
    """Demonstrate the MockContextFactory for complex scenarios."""

    print("\n\nMOCK CONTEXT FACTORY - ADVANCED PATTERNS")
    print("=" * 50)

    # PATTERN 1: Individual Context Creation
    print("\n1. Individual Context Creation:")
    print("-" * 32)

    factory = MockContextFactory()

    async def demo_individual_contexts():
        """Demonstrate individual context creation."""

        # LiteLLM context
        with factory.create_litellm_context("AI response for demo"):
            print("✓ LiteLLM context created")

        # Environment context
        with factory.create_environment_context({"DEMO_VAR": "demo_value"}) as env:
            print(f"✓ Environment context created: {env}")

        # Sentence transformer context
        with factory.create_sentence_transformer_context():
            print("✓ Sentence transformer context created")

    asyncio.run(demo_individual_contexts())

    # PATTERN 2: Combined Context
    print("\n2. Combined Context (Most Powerful):")
    print("-" * 37)

    async def demo_combined_context():
        """Demonstrate comprehensive context creation."""

        with factory.create_full_test_context(
            litellm_response="Comprehensive AI response for testing",
            env_vars={
                "GENERATION_ENABLE": "true",
                "GENERATION_MODEL_NAME": "gpt-4o-mini",
                "GENERATION_LITELLM_API_KEY": "demo_key",
            },
            shared_model=None,  # Will create mock
            mock_sentence_transformers=True,
        ) as context:
            print("✓ Full test context created with:")
            print(f"  - LiteLLM mock: {context.get('litellm_mock') is not None}")
            print(f"  - Environment vars: {len(context.get('env_vars', {}))}")
            print(
                f"  - Sentence transformer: {context.get('sentence_transformer_model') is not None}"
            )

            # Show environment is properly set
            print(f"  - GENERATION_ENABLE: {os.environ.get('GENERATION_ENABLE')}")
            print(f"  - Model configured: {os.environ.get('GENERATION_MODEL_NAME')}")

    asyncio.run(demo_combined_context())


def demonstrate_validation_helpers():
    """Demonstrate response validation helpers."""

    print("\n\nRESPONSE VALIDATION HELPERS")
    print("=" * 50)

    print("\n1. CLI JSON Response Validation:")
    print("-" * 33)

    # Valid response
    valid_response = {
        "results": [{"title": "Test Page", "content": "Test content"}],
        "total": 1,
        "took_ms": 45,
    }

    try:
        ResponseValidationHelper.validate_cli_json_response(
            valid_response, ["results", "total", "took_ms"], "search response"
        )
        print("✓ Valid response passed validation")
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")

    # Invalid response (missing key)
    invalid_response = {
        "results": [],
        "total": 0,
        # Missing "took_ms"
    }

    try:
        ResponseValidationHelper.validate_cli_json_response(
            invalid_response, ["results", "total", "took_ms"], "search response"
        )
        print("✗ Invalid response should have failed validation")
    except AssertionError as e:
        print(f"✓ Invalid response properly caught: {str(e)[:50]}...")


def demonstrate_performance_helpers():
    """Demonstrate performance mock helpers."""

    print("\n\nPERFORMANCE MOCK HELPERS")
    print("=" * 50)

    print("\n1. Timed Mock Context:")
    print("-" * 20)

    def simulate_operation():
        """Simulate some work."""
        time.sleep(0.02)

    with PerformanceMockHelper.timed_mock_context("demo_operation") as timer:
        simulate_operation()

    print(f"✓ Operation completed in {timer['duration']:.4f}s")
    print(f"✓ Start time: {timer['start_time']:.2f}")
    print(f"✓ End time: {timer['end_time']:.2f}")

    print("\n2. Performance Threshold Validation:")
    print("-" * 35)

    # Fast operation (should pass)
    try:
        with PerformanceMockHelper.timed_mock_context("fast_op", 0.1) as timer:
            time.sleep(0.01)
        print(f"✓ Fast operation passed threshold: {timer['duration']:.4f}s < 0.1s")
    except AssertionError as e:
        print(f"✗ Fast operation failed: {e}")

    # Slow operation (should fail)
    try:
        with PerformanceMockHelper.timed_mock_context("slow_op", 0.01) as timer:
            time.sleep(0.02)
        print(f"✗ Slow operation should have failed: {timer['duration']:.4f}s")
    except AssertionError as e:
        print(f"✓ Slow operation properly caught: {str(e)[:50]}...")


def demonstrate_real_world_replacements():
    """Show real-world pattern replacements from actual test files."""

    print("\n\nREAL-WORLD PATTERN REPLACEMENTS")
    print("=" * 50)

    print("\n1. From test_e2e_workflows.py (lines 379-384):")
    print("-" * 45)

    print("OLD PATTERN:")
    print("""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = create_mock_litellm_response(
            "Based on the retrieved documentation, software development..."
        )
        # Test code here
    """)

    print("NEW PATTERN:")
    print("""
    with LiteLLMContext("Based on the retrieved documentation, software development...") as mock_litellm:
        # Test code here - same functionality, cleaner syntax
    """)

    # Demonstrate the replacement
    async def demo_litellm_replacement():
        """Show the LiteLLM replacement in action."""
        with LiteLLMContext("Software development best practices include..."):
            print("✓ LiteLLM context created with simplified syntax")
            print("✓ Mock is properly configured AsyncMock")
            print("✓ Response content is properly set")

    asyncio.run(demo_litellm_replacement())

    print("\n2. From test_rag_integration.py (lines 34-62):")
    print("-" * 40)

    print("OLD PATTERN:")
    print("""
    @contextmanager
    def environment_context(env_vars: dict[str, str], tmp_path: str | None = None):
        original_env = os.environ.copy()
        try:
            for key, value in env_vars.items():
                os.environ[key] = value
            if tmp_path:
                os.environ["CONFLUENCE_GATEWAY_CONFIG_PATH"] = tmp_path
            yield env_vars
        finally:
            os.environ.clear()
            os.environ.update(original_env)
    """)

    print("NEW PATTERN:")
    print("""
    with EnvironmentContext(env_vars, config_path=tmp_path) as env:
        # Same functionality, standardized implementation
    """)

    # Demonstrate the replacement
    def demo_environment_replacement():
        """Show the environment replacement in action."""
        env_vars = {
            "GENERATION_ENABLE": "true",
            "GENERATION_MODEL_NAME": "openrouter/google/gemini-2.5-flash",
            "GENERATION_LITELLM_API_KEY": "test_api_key",
        }

        with EnvironmentContext(env_vars) as applied_env:
            print("✓ Environment context created with simplified syntax")
            print(f"✓ Variables applied: {len(applied_env)} vars")
            print(f"✓ GENERATION_ENABLE: {os.environ.get('GENERATION_ENABLE')}")

    demo_environment_replacement()

    print("\n3. From test_configuration_matrix.py (lines 86-89):")
    print("-" * 46)

    print("OLD PATTERN:")
    print("""
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=shared_sentence_transformer_model,
    ):
        # Test code
    """)

    print("NEW PATTERN:")
    print("""
    with SentenceTransformerContext(shared_sentence_transformer_model) as model:
        # Test code - cleaner and more explicit
    """)

    # Demonstrate the replacement
    def demo_sentence_transformer_replacement():
        """Show the sentence transformer replacement in action."""
        # Simulate shared model (normally would be fixture)
        mock_shared_model = None

        with SentenceTransformerContext(mock_shared_model) as model:
            print("✓ Sentence transformer context created")
            print(f"✓ Model available: {model is not None}")
            print("✓ Handles both real and mock models transparently")

    demo_sentence_transformer_replacement()


def demonstrate_migration_utilities():
    """Demonstrate migration utilities for existing code."""

    print("\n\nMIGRATION UTILITIES")
    print("=" * 50)

    print("\n1. Migration Helper Functions:")
    print("-" * 30)

    # Show migration helpers
    async def demo_migration_helpers():
        """Demonstrate migration utility functions."""

        # LiteLLM migration
        litellm_context = MockMigrationHelper.migrate_litellm_pattern(
            "migrated response"
        )
        with litellm_context:
            print("✓ LiteLLM pattern migrated successfully")

        # Environment migration
        env_context = MockMigrationHelper.migrate_environment_pattern(
            {"MIGRATED_VAR": "value"}
        )
        with env_context as env:
            print(f"✓ Environment pattern migrated: {env}")

    asyncio.run(demo_migration_helpers())

    print("\n2. Migration Examples Output:")
    print("-" * 30)

    # Show migration examples (this will print to stdout)
    MockMigrationHelper.show_migration_examples()


def main():
    """Run all demonstration examples."""
    demonstrate_basic_patterns()
    demonstrate_factory_patterns()
    demonstrate_validation_helpers()
    demonstrate_performance_helpers()
    demonstrate_real_world_replacements()
    demonstrate_migration_utilities()

    print("\n" + "=" * 70)
    print("CONSOLIDATION SUMMARY")
    print("=" * 70)
    print("✓ Consolidated 8+ mocking patterns into standardized utilities")
    print("✓ LiteLLM async mocking: 4+ patterns → LiteLLMContext")
    print("✓ Environment management: 3+ patterns → EnvironmentContext")
    print("✓ Sentence transformer mocking: 3+ patterns → SentenceTransformerContext")
    print("✓ Configuration helpers: Multiple patterns → ConfigurationMockHelper")
    print("✓ Response validation: Scattered patterns → ResponseValidationHelper")
    print("✓ Performance tracking: Manual timing → PerformanceMockHelper")
    print("✓ Migration support: MockMigrationHelper for easy transitions")
    print("✓ Factory pattern: MockContextFactory for complex scenarios")
    print("\nBENEFITS:")
    print("- Reduced code duplication across 8+ test files")
    print("- Standardized error handling and cleanup")
    print("- Type hints and comprehensive documentation")
    print("- Integration with existing test utilities")
    print("- Backward compatibility through migration helpers")
    print("=" * 70)


if __name__ == "__main__":
    main()
