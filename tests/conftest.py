import os
import socket
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

# Import testing mode detection
from confluence_gateway.core.config import get_testing_mode, should_use_memory_mode

# Import shared embedding optimization fixtures
from tests.fixtures.shared_embedding import (
    embedding_service_with_shared_model,
    get_shared_model_thread_safe,
    inject_shared_model_into_provider,
    log_embedding_operation,
    mock_sentence_transformer_with_shared_model,
    performance_tracker,
    shared_embedding_provider,
    shared_sentence_transformer_model,
)

# Configure vector database based on testing mode
# Local testing: use memory mode for isolation and speed
# CI testing: use file storage with caching for better performance across test runs

# Global test storage directory for CI mode
_test_storage_dir = None


def _setup_test_storage_environment():
    """Configure test storage environment based on testing mode."""
    global _test_storage_dir

    use_memory = should_use_memory_mode()

    if use_memory:
        # Local testing - use memory mode
        os.environ["QDRANT_URL"] = ":memory:"
        os.environ["CHROMA_PERSIST_PATH"] = ""
        os.environ["CHROMA_HOST"] = ""
        os.environ["CHROMA_PORT"] = ""
        print("\n🧪 Test configuration: LOCAL mode (memory storage)")
    else:
        # CI testing - use file storage with caching
        if _test_storage_dir is None:
            _test_storage_dir = Path(
                tempfile.mkdtemp(prefix="confluence_gateway_ci_test_")
            )

        # Set up file-based storage for CI
        qdrant_path = _test_storage_dir / "qdrant_data"
        chroma_path = _test_storage_dir / "chroma_data"

        # Create directories
        qdrant_path.mkdir(parents=True, exist_ok=True)
        chroma_path.mkdir(parents=True, exist_ok=True)

        # Configure file storage
        os.environ["QDRANT_LOCAL_PATH"] = str(qdrant_path)
        os.environ["CHROMA_PERSIST_PATH"] = str(chroma_path)

        # Remove memory mode settings if they exist
        os.environ.pop("QDRANT_URL", None)
        os.environ.pop("CHROMA_HOST", None)
        os.environ.pop("CHROMA_PORT", None)

        print(f"\n🏗️  Test configuration: CI mode (file storage at {_test_storage_dir})")

    # Common test settings
    if "VECTOR_DB_TYPE" not in os.environ:
        os.environ["VECTOR_DB_TYPE"] = "qdrant"  # Use qdrant as default for tests

    # Set required embedding dimension for vector DB
    if "VECTOR_DB_EMBEDDING_DIMENSION" not in os.environ:
        os.environ["VECTOR_DB_EMBEDDING_DIMENSION"] = (
            "384"  # Default dimension for all-MiniLM-L6-v2
        )

    # Force sentence-transformers provider for consistency with tests
    os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"
    os.environ["EMBEDDING_MODEL_NAME"] = "all-MiniLM-L6-v2"
    os.environ["EMBEDDING_DIMENSION"] = "384"
    os.environ["EMBEDDING_DEVICE"] = "cpu"


# Set up test storage environment at import time
_setup_test_storage_environment()


@pytest.fixture(scope="session", autouse=True)
def setup_shared_sentence_transformer_optimization(
    shared_sentence_transformer_model,
    shared_embedding_provider,
    embedding_service_with_shared_model,
    performance_tracker,
):
    """
    Auto-use session fixture that sets up shared sentence-transformer optimization.

    This fixture ensures that:
    1. Shared sentence-transformer model is available for the session
    2. Shared embedding provider and service are initialized
    3. Performance tracking is enabled across all optimization layers
    4. Proper cleanup occurs at session end

    The autouse=True ensures this runs automatically for all tests that might
    need embedding functionality, without requiring explicit fixture requests.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Log optimization setup with detailed status
    optimization_status = {
        "shared_model": shared_sentence_transformer_model is not None,
        "shared_provider": shared_embedding_provider is not None,
        "shared_service": embedding_service_with_shared_model is not None,
    }

    # Log testing mode for debugging
    testing_mode = get_testing_mode()
    storage_mode = "memory" if should_use_memory_mode() else "file"

    if optimization_status["shared_model"]:
        logger.info(
            "Session-scoped sentence-transformer optimization ENABLED. "
            f"Status: Model={optimization_status['shared_model']}, "
            f"Provider={optimization_status['shared_provider']}, "
            f"Service={optimization_status['shared_service']}"
        )
        print(
            f"\n🚀 Shared embedding optimization active ({testing_mode} mode, {storage_mode} storage) - "
            "significant performance improvements expected for embedding tests"
        )
    else:
        logger.info(
            "sentence-transformers not available or failed to load. "
            "Tests will use fallback behavior (dev mode stubs)."
        )
        print(
            f"\n⚠️  Shared embedding optimization unavailable ({testing_mode} mode, {storage_mode} storage) - "
            "tests will use standard initialization (slower)"
        )

    yield optimization_status

    # Session cleanup logging with performance summary
    logger.info(
        "Shared sentence-transformer optimization session complete. "
        "Performance metrics logged by performance_tracker fixture."
    )


@pytest.fixture(scope="session")
def api_server(shared_sentence_transformer_model):
    """Start FastAPI server for testing with dynamic port allocation and health checking"""
    # Find available port dynamically
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    # Get current environment and add our test-specific overrides
    env = os.environ.copy()

    # Apply test storage configuration based on testing mode
    if should_use_memory_mode():
        env.update(
            {
                "QDRANT_URL": ":memory:",
                "CHROMA_PERSIST_PATH": "",
                "CHROMA_HOST": "",
                "CHROMA_PORT": "",
            }
        )
    else:
        # CI mode - use file storage (paths already set in global setup)
        env.update(
            {
                "QDRANT_LOCAL_PATH": env.get("QDRANT_LOCAL_PATH", ""),
                "CHROMA_PERSIST_PATH": env.get("CHROMA_PERSIST_PATH", ""),
            }
        )

    # Common test environment settings
    env.update(
        {
            "VECTOR_DB_TYPE": env.get("VECTOR_DB_TYPE", "qdrant"),
            "VECTOR_DB_EMBEDDING_DIMENSION": env.get(
                "VECTOR_DB_EMBEDDING_DIMENSION", "384"
            ),
            # Force sentence-transformers provider for consistency with tests
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
            "EMBEDDING_DEVICE": "cpu",
        }
    )

    # Start server with dynamic port and test environment variables
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "confluence_gateway.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
    )

    # Health check polling instead of sleep
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(30):  # 30 second timeout
        try:
            response = httpx.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                break
        except (httpx.RequestError, httpx.TimeoutException):
            time.sleep(1)
    else:
        # Server failed to start within timeout
        process.terminate()
        process.wait()
        raise RuntimeError("API server failed to start within 30 seconds")

    # Additional startup time for model loading and initialization
    # Optimize delay based on whether shared model is available and testing mode
    testing_mode = get_testing_mode()

    if shared_sentence_transformer_model is not None:
        if testing_mode == "ci":
            print(
                "API server started with shared model optimization (CI mode), allowing 3 seconds for initialization..."
            )
            time.sleep(3)  # Slightly longer for CI with file storage
        else:
            print(
                "API server started with shared model optimization (local mode), allowing 2 seconds for initialization..."
            )
            time.sleep(2)  # Reduced delay since model is pre-loaded
    else:
        if testing_mode == "ci":
            print(
                "API server started (CI mode), allowing 7 seconds for model initialization..."
            )
            time.sleep(7)  # Longer delay for CI without optimization
        else:
            print(
                "API server started (local mode), allowing 5 seconds for model initialization..."
            )
            time.sleep(5)  # Full delay for non-optimized startup

    yield base_url

    # Clean termination
    process.terminate()
    process.wait()


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_storage():
    """Clean up test storage directories after all tests complete."""
    global _test_storage_dir

    yield  # Let all tests run first

    # Clean up CI test storage if it was created
    if _test_storage_dir and _test_storage_dir.exists():
        import shutil

        try:
            shutil.rmtree(_test_storage_dir)
            print(f"\n🧹 Cleaned up test storage: {_test_storage_dir}")
        except Exception as e:
            print(
                f"\n⚠️  Warning: Could not clean up test storage {_test_storage_dir}: {e}"
            )
        finally:
            _test_storage_dir = None


@pytest.fixture
def api_client(api_server):
    """API client fixture with extended timeout for AI operations"""
    return httpx.Client(
        base_url=api_server,
        timeout=httpx.Timeout(30.0),  # 30 second timeout for AI model operations
    )


@pytest.fixture
def cli_command_base():
    """CLI command base prefix"""
    return ["uv", "run", "confluence-gateway"]


@pytest.fixture
def embedding_model_context(shared_sentence_transformer_model):
    """
    Provides access to the shared sentence-transformer model for tests.

    This fixture gives tests direct access to the session-scoped model
    for validation and testing purposes.

    Returns:
        dict: Context containing model information and availability status
    """
    return {
        "model": shared_sentence_transformer_model,
        "available": shared_sentence_transformer_model is not None,
        "model_name": "all-MiniLM-L6-v2",  # Default test model
        "dimension": 384,  # Default dimension
        "thread_safe_access": lambda: get_shared_model_thread_safe(
            shared_sentence_transformer_model
        ),
    }


@pytest.fixture
def optimized_embedding_context(
    shared_sentence_transformer_model,
    shared_embedding_provider,
    embedding_service_with_shared_model,
):
    """
    Comprehensive fixture providing access to all shared embedding optimization layers.

    This fixture provides a convenient way for tests to access any level of the
    embedding optimization stack, from raw model to high-level service.

    Returns:
        dict: Complete embedding optimization context
    """
    context = {
        "model": shared_sentence_transformer_model,
        "provider": shared_embedding_provider,
        "service": embedding_service_with_shared_model,
        "available": {
            "model": shared_sentence_transformer_model is not None,
            "provider": shared_embedding_provider is not None,
            "service": embedding_service_with_shared_model is not None,
        },
        "inject_model": inject_shared_model_into_provider,
        "log_operation": log_embedding_operation,
    }

    # Log context availability for debugging
    available_layers = [k for k, v in context["available"].items() if v]
    testing_mode = get_testing_mode()
    storage_mode = "memory" if should_use_memory_mode() else "file"
    print(
        f"Optimized embedding context ({testing_mode} mode, {storage_mode} storage) - available layers: {available_layers}"
    )

    return context


@pytest.fixture
def embedding_performance_tracker():
    """
    Fixture for tracking embedding operation performance in individual tests.

    This fixture provides utilities for measuring and logging the performance
    impact of embedding operations within specific tests.

    Returns:
        dict: Performance tracking utilities
    """
    start_times = {}

    def start_operation(operation_name: str) -> None:
        """Start timing an embedding operation."""
        start_times[operation_name] = time.time()

    def end_operation(operation_name: str) -> float:
        """End timing an operation and log the duration."""
        if operation_name not in start_times:
            raise ValueError(f"Operation '{operation_name}' was not started")

        duration = time.time() - start_times[operation_name]
        log_embedding_operation(operation_name, duration)
        del start_times[operation_name]
        return duration

    return {
        "start_operation": start_operation,
        "end_operation": end_operation,
        "log_operation": log_embedding_operation,
    }
