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

# Import shared embedding optimization fixtures
from tests.fixtures.shared_embedding import (
    embedding_service_with_shared_model,
    get_shared_model_thread_safe,
    inject_shared_model_into_provider,
    log_embedding_operation,
    performance_tracker,
    shared_embedding_provider,
    shared_sentence_transformer_model,
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
def optimized_embedding_provider(
    shared_sentence_transformer_model,
    provider_config: dict[str, Any],
):
    """Create an optimized embedding provider for integration tests.

    This fixture combines the shared model optimization with the provider
    configuration matrix, allowing integration tests to benefit from
    performance improvements while testing different provider combinations.

    Args:
        shared_sentence_transformer_model: Session-scoped shared model
        provider_config: Current provider configuration from matrix

    Returns:
        Optimized embedding provider instance or None if not applicable
    """
    embedding_provider = provider_config.get("EMBEDDING_PROVIDER")

    # Only optimize sentence-transformers providers
    if embedding_provider != "sentence-transformers":
        return None

    if shared_sentence_transformer_model is None:
        return None

    try:
        import time

        from confluence_gateway.adapters.embedding.sentence_transformer import (
            SentenceTransformerProvider,
        )
        from confluence_gateway.core.config import EmbeddingConfig

        start_time = time.time()

        # Create provider with configuration from matrix
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name=provider_config.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            device=provider_config.get("EMBEDDING_DEVICE", "cpu"),
            dimension=int(provider_config.get("EMBEDDING_DIMENSION", "384")),
        )
        provider = SentenceTransformerProvider(config)

        # Inject shared model for optimization
        success = inject_shared_model_into_provider(
            provider, shared_sentence_transformer_model
        )

        # Log performance metrics
        creation_time = time.time() - start_time
        log_embedding_operation("provider_creation_with_injection", creation_time)

        if success:
            print(
                f"Integration test using optimized embedding provider with shared model (created in {creation_time:.3f}s)"
            )
            return provider
        else:
            print("Warning: Could not inject shared model, using standard provider")
            return provider

    except ImportError:
        print("Warning: SentenceTransformerProvider not available for optimization")
        return None
    except Exception as e:
        print(f"Warning: Could not create optimized embedding provider: {e}")
        return None


@pytest.fixture
def integration_embedding_service(
    shared_sentence_transformer_model,
    provider_config: dict[str, Any],
):
    """Create an optimized EmbeddingService for integration tests.

    This fixture provides a higher-level EmbeddingService instance that
    benefits from shared model optimization when using sentence-transformers.

    Args:
        shared_sentence_transformer_model: Session-scoped shared model
        provider_config: Current provider configuration from matrix

    Returns:
        EmbeddingService instance with optimization if applicable
    """
    embedding_provider = provider_config.get("EMBEDDING_PROVIDER")

    # For non-sentence-transformers providers, return None
    if embedding_provider != "sentence-transformers":
        return None

    try:
        import time

        from confluence_gateway.adapters.embedding.sentence_transformer import (
            SentenceTransformerProvider,
        )
        from confluence_gateway.core.config import EmbeddingConfig
        from confluence_gateway.services.embedding import EmbeddingService

        start_time = time.time()

        # Create provider with shared model optimization
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name=provider_config.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            device=provider_config.get("EMBEDDING_DEVICE", "cpu"),
            dimension=int(provider_config.get("EMBEDDING_DIMENSION", "384")),
        )
        provider = SentenceTransformerProvider(config)

        # Inject shared model if available
        if shared_sentence_transformer_model is not None:
            success = inject_shared_model_into_provider(
                provider, shared_sentence_transformer_model
            )
            if success:
                print("Integration EmbeddingService using shared model optimization")

        # Create service with optimized provider
        service = EmbeddingService(provider=provider)

        # Log performance metrics
        creation_time = time.time() - start_time
        log_embedding_operation("embedding_service_creation", creation_time)

        print(f"Created integration EmbeddingService in {creation_time:.3f}s")
        return service

    except ImportError:
        print("Warning: EmbeddingService components not available")
        return None
    except Exception as e:
        print(f"Warning: Could not create integration EmbeddingService: {e}")
        return None


@pytest.fixture
def integration_embedding_context(
    shared_sentence_transformer_model,
    optimized_embedding_provider,
    integration_embedding_service,
    provider_config: dict[str, Any],
):
    """Comprehensive embedding context for integration tests.

    This fixture provides integration tests with complete access to the
    embedding optimization stack, tailored for the current provider matrix.

    Returns:
        dict: Complete integration embedding context
    """
    context = {
        "shared_model": shared_sentence_transformer_model,
        "optimized_provider": optimized_embedding_provider,
        "optimized_service": integration_embedding_service,
        "provider_config": provider_config,
        "optimization_available": {
            "model": shared_sentence_transformer_model is not None,
            "provider": optimized_embedding_provider is not None,
            "service": integration_embedding_service is not None,
        },
        "utilities": {
            "inject_model": inject_shared_model_into_provider,
            "log_operation": log_embedding_operation,
            "thread_safe_access": lambda: get_shared_model_thread_safe(
                shared_sentence_transformer_model
            ),
        },
    }

    # Log context status for debugging
    optimizations = [k for k, v in context["optimization_available"].items() if v]
    provider_type = provider_config.get("EMBEDDING_PROVIDER", "none")

    print(
        f"Integration embedding context: {provider_type} provider, "
        f"optimizations available: {optimizations}"
    )

    return context


@pytest.fixture
def isolated_test_environment(
    clean_environment: dict[str, str],
    config_injection: Any,
    resource_cleanup: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Provide a completely isolated test environment with embedding optimization support.

    Args:
        clean_environment: Clean environment fixture
        config_injection: Configuration injection utility
        resource_cleanup: Resource cleanup utility

    Yields:
        Dictionary with utility functions for test isolation and performance tracking
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(
        "Setting up isolated test environment with embedding optimization support"
    )

    environment_context = {
        "inject_config": config_injection,
        "cleanup": resource_cleanup,
        "original_env": clean_environment,
        "log_operation": log_embedding_operation,
    }

    yield environment_context

    # Ensure cleanup happens with performance logging
    logger.info("Cleaning up isolated test environment")
    resource_cleanup["cleanup_storage"]()


@pytest.fixture(scope="session")
def integration_optimization_summary(
    shared_sentence_transformer_model,
    shared_embedding_provider,
):
    """Session-scoped fixture that provides optimization summary for integration tests.

    This fixture runs once per session and provides a summary of available
    optimizations that integration tests can leverage.

    Returns:
        dict: Summary of optimization capabilities
    """
    import logging

    logger = logging.getLogger(__name__)

    summary = {
        "session_optimization_enabled": shared_sentence_transformer_model is not None,
        "shared_provider_available": shared_embedding_provider is not None,
        "expected_performance_improvement": "significant"
        if shared_sentence_transformer_model
        else "none",
        "recommendation": (
            "Integration tests will benefit from shared model optimization"
            if shared_sentence_transformer_model
            else "Consider installing sentence-transformers for better test performance"
        ),
    }

    logger.info(
        f"Integration test optimization summary: "
        f"enabled={summary['session_optimization_enabled']}, "
        f"improvement={summary['expected_performance_improvement']}"
    )

    print("\n" + "=" * 50)
    print("INTEGRATION TEST OPTIMIZATION SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 50)

    return summary


@pytest.fixture(scope="session", autouse=True)
def integration_performance_tracking(
    integration_optimization_summary,
):
    """Track performance improvements in integration tests.

    This session-scoped fixture monitors the performance impact of shared
    embedding optimization specifically in integration test contexts.

    Args:
        integration_optimization_summary: Summary of available optimizations
    """
    import time

    print("\n" + "=" * 50)
    print("INTEGRATION TESTS - Shared Embedding Optimization")
    print("=" * 50)

    if integration_optimization_summary["session_optimization_enabled"]:
        print("🚀 OPTIMIZATION ACTIVE: Shared sentence-transformer model available")
        print("Expected significant performance improvements for embedding operations")
    else:
        print("⚠️  OPTIMIZATION UNAVAILABLE: Using standard initialization")
        print("Tests will run but may be slower for embedding operations")

    session_start = time.time()

    yield

    session_duration = time.time() - session_start
    print(f"\nIntegration test session completed in {session_duration:.2f}s")
    if integration_optimization_summary["session_optimization_enabled"]:
        print("🚀 Shared embedding model optimization was active throughout session")
    else:
        print("⚠️  No shared model optimization (sentence-transformers unavailable)")
    print("=" * 50)
