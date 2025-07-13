import subprocess
import pytest


def test_generate_answer_command():
    """Test generate answer command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "generate", "answer", "What is Confluence?"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "answer" in data
    assert "sources" in data