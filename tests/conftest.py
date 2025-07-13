import os
import socket
import subprocess
import time
from collections.abc import Generator

import httpx
import pytest

# Force vector database memory mode for tests to prevent file locking issues
# and make tests more reliable and isolated
os.environ["QDRANT_URL"] = ":memory:"
os.environ["CHROMA_PERSIST_PATH"] = ""
os.environ["CHROMA_HOST"] = ""
os.environ["CHROMA_PORT"] = ""
# Ensure vector DB is enabled for testing with memory storage
if "VECTOR_DB_TYPE" not in os.environ:
    os.environ["VECTOR_DB_TYPE"] = "qdrant"  # Use qdrant with memory mode by default
# Set required embedding dimension for vector DB
if "VECTOR_DB_EMBEDDING_DIMENSION" not in os.environ:
    os.environ["VECTOR_DB_EMBEDDING_DIMENSION"] = (
        "384"  # Default dimension for all-MiniLM-L6-v2
    )


@pytest.fixture(scope="session")
def api_server():
    """Start FastAPI server for testing with dynamic port allocation and health checking"""
    # Find available port dynamically
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    # Get current environment and add our test-specific overrides
    env = os.environ.copy()
    env.update(
        {
            "QDRANT_URL": ":memory:",
            "CHROMA_PERSIST_PATH": "",
            "CHROMA_HOST": "",
            "CHROMA_PORT": "",
            "VECTOR_DB_TYPE": env.get("VECTOR_DB_TYPE", "qdrant"),
            "VECTOR_DB_EMBEDDING_DIMENSION": env.get(
                "VECTOR_DB_EMBEDDING_DIMENSION", "384"
            ),
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
    print("API server started, allowing 5 seconds for model initialization...")
    time.sleep(5)

    yield base_url

    # Clean termination
    process.terminate()
    process.wait()


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
