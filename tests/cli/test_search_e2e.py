import subprocess
import pytest


def test_search_text_command():
    """Test search text command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "search", "text", "test"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "results" in data
    assert "total" in data


def test_search_semantic_command():
    """Test search semantic command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "search", "semantic", "test query"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "results" in data
    assert "query" in data


def test_search_cql_command():
    """Test search cql command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "search", "cql", "text ~ test"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "results" in data


def test_search_hybrid_command():
    """Test search hybrid command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "search", "hybrid", "test"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "results" in data