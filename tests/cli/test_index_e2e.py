import subprocess
import pytest
import json


def parse_cli_json_output(output: str) -> dict:
    """Parse JSON from CLI output that may contain info messages before JSON."""
    lines = output.strip().split('\n')
    
    # Find the first line that starts with '{' and collect all subsequent lines
    json_started = False
    json_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Start collecting JSON when we find the opening brace
        if not json_started and stripped_line.startswith('{'):
            json_started = True
            json_lines.append(line)
        elif json_started:
            # Continue collecting lines that are part of the JSON
            json_lines.append(line)
    
    if json_lines:
        # Join all JSON lines and parse
        json_text = '\n'.join(json_lines)
        return json.loads(json_text)
    
    # If no JSON found, try parsing the entire output as fallback
    return json.loads(output.strip())


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
    # CLI outputs JSON format (may have info messages before JSON)
    data = parse_cli_json_output(result.stdout)
    assert "status" in data