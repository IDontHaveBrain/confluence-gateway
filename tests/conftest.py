import pytest
import subprocess
import time
import httpx
from typing import Generator


@pytest.fixture(scope="session")
def api_server():
    """Start FastAPI server for testing"""
    # Start server (background)
    process = subprocess.Popen([
        "uv", "run", "uvicorn", 
        "confluence_gateway.api.app:app",
        "--host", "127.0.0.1",
        "--port", "8001"  # Test port
    ])
    
    # Wait for server to start
    time.sleep(3)
    
    yield "http://127.0.0.1:8001"
    
    # Terminate server
    process.terminate()
    process.wait()


@pytest.fixture
def api_client(api_server):
    """API client fixture"""
    return httpx.Client(base_url=api_server)


@pytest.fixture
def cli_command_base():
    """CLI command base prefix"""
    return ["uv", "run", "confluence-gateway"]