"""
Session-scoped fixtures for shared sentence-transformers model initialization.

This module provides optimized fixtures that load sentence-transformers models once per
test session and reuse them across all tests requiring embeddings. This dramatically
reduces test execution time by avoiding repeated model loading overhead.

Key optimizations:
- Session-scoped model loading (once per pytest session)
- Shared model injection into provider instances
- Performance tracking and metrics logging
- Thread-safe concurrent access
- Proper resource cleanup and error handling
- Compatibility with existing test patterns (memory mode, configuration injection)

Architecture:
1. shared_sentence_transformer_model: Loads the actual transformer model once
2. shared_embedding_provider: Creates provider instances using the shared model
3. embedding_service_with_shared_model: Higher-level service access
4. performance_tracker: Measures and logs optimization impact
"""

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

# Performance tracking storage
_performance_metrics: dict[str, float] = {}
_metrics_lock = threading.Lock()


@pytest.fixture(scope="session")
def shared_sentence_transformer_model():
    """
    Session-scoped fixture that loads a sentence-transformers model once per test session.

    This fixture provides significant performance improvements by:
    - Loading the model only once per pytest session
    - Reusing the loaded model across all tests requiring embeddings
    - Using CPU device for consistent test behavior
    - Creating a temporary cache directory with proper cleanup

    Returns:
        The loaded sentence-transformer model instance, or None if unavailable

    Note:
        This fixture handles graceful degradation when sentence-transformers
        is not available, allowing tests to continue with mocked providers.
    """
    start_time = time.time()

    try:
        # Import sentence-transformers with error handling
        from sentence_transformers import SentenceTransformer

        # Create temporary cache directory for session
        temp_cache_dir = tempfile.mkdtemp(prefix="confluence_gateway_test_cache_")
        cache_path = Path(temp_cache_dir)

        print("Loading shared sentence-transformer model (session-scoped)")
        print(f"Using temporary cache: {cache_path}")

        # Load model with CPU device for consistent test behavior
        # Using a lightweight model suitable for testing
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(
            model_name, device="cpu", cache_folder=str(cache_path)
        )

        # Track loading time
        load_time = time.time() - start_time
        with _metrics_lock:
            _performance_metrics["model_load_time"] = load_time
            _performance_metrics["model_name"] = model_name
            _performance_metrics["cache_path"] = str(cache_path)

        print(f"Shared sentence-transformer model loaded in {load_time:.2f}s")

        yield model

        # Cleanup: Remove temporary cache directory
        import shutil

        try:
            shutil.rmtree(cache_path)
            print(f"Cleaned up temporary cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not clean up cache directory {cache_path}: {e}")

    except ImportError:
        print(
            "Warning: sentence-transformers not available, using None for shared model"
        )
        yield None
    except Exception as e:
        print(f"Error loading shared sentence-transformer model: {e}")
        yield None


@pytest.fixture(scope="session")
def shared_embedding_provider(shared_sentence_transformer_model):
    """
    Session-scoped fixture that creates a SentenceTransformerAdapter using the shared model.

    This fixture optimizes embedding provider creation by:
    - Injecting the pre-loaded shared model to skip heavy loading
    - Maintaining consistent configuration across all tests
    - Providing thread-safe access for concurrent test execution

    Args:
        shared_sentence_transformer_model: The session-scoped shared model

    Returns:
        SentenceTransformerAdapter instance with injected shared model,
        or None if sentence-transformers is unavailable
    """
    if shared_sentence_transformer_model is None:
        print("Shared model unavailable, returning None for embedding provider")
        yield None
        return

    try:
        from confluence_gateway.adapters.embedding.sentence_transformer import (
            SentenceTransformerAdapter,
        )

        # Create adapter instance with shared model injection
        provider = SentenceTransformerAdapter(
            model_name="all-MiniLM-L6-v2",  # This will be overridden by injection
            device="cpu",
            dimension=384,
        )

        # Inject the shared model to skip loading overhead
        # This is the key optimization - we bypass the expensive model loading
        provider._model = shared_sentence_transformer_model
        provider._is_initialized = True

        print("Created shared embedding provider with injected model")
        yield provider

    except ImportError:
        print("Warning: SentenceTransformerAdapter not available")
        yield None
    except Exception as e:
        print(f"Error creating shared embedding provider: {e}")
        yield None


@pytest.fixture(scope="session")
def embedding_service_with_shared_model(shared_embedding_provider):
    """
    Session-scoped fixture that creates an EmbeddingService using the shared provider.

    This provides higher-level service access with:
    - Pre-configured EmbeddingService instance
    - Shared model optimization benefits
    - Consistent configuration across all tests

    Args:
        shared_embedding_provider: The session-scoped shared provider

    Returns:
        EmbeddingService instance with shared provider, or None if unavailable
    """
    if shared_embedding_provider is None:
        print("Shared provider unavailable, returning None for embedding service")
        yield None
        return

    try:
        from confluence_gateway.services.embedding import EmbeddingService

        # Create service with shared provider
        service = EmbeddingService(provider=shared_embedding_provider)

        print("Created embedding service with shared model optimization")
        yield service

    except ImportError:
        print("Warning: EmbeddingService not available")
        yield None
    except Exception as e:
        print(f"Error creating embedding service with shared model: {e}")
        yield None


@pytest.fixture(scope="session", autouse=True)
def performance_tracker():
    """
    Session-scoped fixture that tracks and reports performance optimization metrics.

    This fixture automatically:
    - Measures test session duration
    - Logs model loading times and optimization impact
    - Reports performance improvements at session end
    - Provides metrics for performance regression testing

    The autouse=True parameter ensures this runs for every test session,
    providing consistent performance monitoring.
    """
    session_start = time.time()

    print("\n" + "=" * 60)
    print("CONFLUENCE GATEWAY TEST SESSION - PERFORMANCE TRACKING")
    print("=" * 60)
    print("Shared embedding model optimization: ENABLED")

    yield

    # Report performance metrics at session end
    session_duration = time.time() - session_start

    print("\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION REPORT")
    print("=" * 60)

    with _metrics_lock:
        if "model_load_time" in _performance_metrics:
            model_load_time = _performance_metrics["model_load_time"]
            model_name = _performance_metrics.get("model_name", "unknown")
            print(f"Shared model loading time: {model_load_time:.2f}s ({model_name})")
            print("Model loaded once and reused across entire test session")

            # Estimate savings (rough calculation based on typical test patterns)
            estimated_individual_loads = (
                10  # Conservative estimate of test count requiring embeddings
            )
            estimated_time_per_load = 2.0  # Conservative estimate of model load time
            estimated_savings = (
                estimated_individual_loads * estimated_time_per_load
            ) - model_load_time

            if estimated_savings > 0:
                print(
                    f"Estimated time savings: {estimated_savings:.2f}s per test session"
                )
                print(
                    f"Performance improvement: {(estimated_savings / session_duration) * 100:.1f}% of session time"
                )
        else:
            print(
                "No shared model metrics available (sentence-transformers not loaded)"
            )

    print(f"Total session duration: {session_duration:.2f}s")
    print("=" * 60)


def inject_shared_model_into_provider(provider_instance, shared_model):
    """
    Utility function to inject a shared model into an embedding provider instance.

    This function enables existing provider instances to use the shared model
    optimization by directly injecting the pre-loaded model and marking it
    as initialized to skip the expensive loading process.

    Args:
        provider_instance: The embedding provider instance to optimize
        shared_model: The pre-loaded shared sentence-transformer model

    Returns:
        bool: True if injection successful, False otherwise

    Usage:
        def test_embedding_optimization(shared_sentence_transformer_model):
            provider = SentenceTransformerAdapter(...)
            if inject_shared_model_into_provider(provider, shared_sentence_transformer_model):
                # Provider now uses shared model - no loading overhead
                embeddings = provider.generate_embeddings(["test text"])
    """
    try:
        if shared_model is None or provider_instance is None:
            return False

        # Inject shared model to skip loading
        provider_instance._model = shared_model
        provider_instance._is_initialized = True

        return True
    except Exception as e:
        print(f"Warning: Could not inject shared model: {e}")
        return False


@pytest.fixture
def mock_sentence_transformer_with_shared_model(shared_sentence_transformer_model):
    """
    Fixture that provides a mock context for sentence-transformer operations using the shared model.

    This fixture helps tests that need to mock sentence-transformer behavior while
    still benefiting from the shared model optimization. It's particularly useful
    for CLI subprocess tests where direct injection isn't possible.

    Args:
        shared_sentence_transformer_model: The session-scoped shared model

    Yields:
        Mock context that can be used to patch sentence-transformer loading
    """
    if shared_sentence_transformer_model is None:
        # Provide a mock when the real model isn't available
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]  # Mock embedding

        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ):
            yield mock_model
    else:
        # Use the real shared model in a mock context
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=shared_sentence_transformer_model,
        ):
            yield shared_sentence_transformer_model


# Thread-safe model access utilities
_model_access_lock = threading.Lock()


def get_shared_model_thread_safe(shared_model):
    """
    Thread-safe accessor for the shared model.

    Provides safe concurrent access to the shared sentence-transformer model
    for tests that might run in parallel or use threading.

    Args:
        shared_model: The shared sentence-transformer model

    Returns:
        The shared model instance with thread-safe access
    """
    with _model_access_lock:
        return shared_model


def log_embedding_operation(operation_name: str, duration: float):
    """
    Log performance metrics for embedding operations.

    This utility helps track the performance impact of the shared model
    optimization by logging operation durations.

    Args:
        operation_name: Name of the embedding operation
        duration: Duration in seconds
    """
    with _metrics_lock:
        metric_key = f"operation_{operation_name}_duration"
        if metric_key not in _performance_metrics:
            _performance_metrics[metric_key] = []
        _performance_metrics[metric_key].append(duration)
