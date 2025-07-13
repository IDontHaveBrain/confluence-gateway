import subprocess
import pytest


def test_spaces_list_command():
    """Test spaces list command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "spaces", "list"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # CLI now outputs JSON format
    import json
    data = json.loads(result.stdout)
    assert "spaces" in data
    assert "pagination" in data


def test_spaces_list_all_command():
    """Test spaces list --all command"""
    result = subprocess.run([
        "uv", "run", "confluence-gateway", "spaces", "list", "--all"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0


def test_spaces_info_command():
    """Test spaces info command (requires actual space key)"""
    # First get actual space key from spaces list
    list_result = subprocess.run([
        "uv", "run", "confluence-gateway", "spaces", "list"
    ], capture_output=True, text=True)
    
    if list_result.returncode == 0 and list_result.stdout:
        # Extract first space key from JSON output
        import json
        data = json.loads(list_result.stdout)
        if data["spaces"]:
            space_key = data["spaces"][0]["key"]
            info_result = subprocess.run([
                "uv", "run", "confluence-gateway", "spaces", "info", space_key
            ], capture_output=True, text=True)
            assert info_result.returncode == 0
            space_data = json.loads(info_result.stdout)
            assert "key" in space_data
            assert "name" in space_data