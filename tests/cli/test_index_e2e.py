import subprocess
import pytest


def test_index_trigger_command():
    """Test index trigger command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "index", "trigger"
    ], capture_output=True, text=True)
    
    # Indexing can take a long time, so just check successful start
    assert result.returncode == 0


def test_index_status_command():
    """Test index status command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "index", "status"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "status" in data