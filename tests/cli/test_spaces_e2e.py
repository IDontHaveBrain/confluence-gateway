import json
import subprocess

import pytest


def parse_cli_json_output(output: str) -> dict:
    """Parse JSON from CLI output that may contain info messages before JSON."""
    lines = output.strip().split("\n")

    # Find the first line that starts with '{' and collect all subsequent lines
    json_started = False
    json_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Start collecting JSON when we find the opening brace
        if not json_started and stripped_line.startswith("{"):
            json_started = True
            json_lines.append(line)
        elif json_started:
            # Continue collecting lines that are part of the JSON
            json_lines.append(line)

    if json_lines:
        # Join all JSON lines and parse
        json_text = "\n".join(json_lines)
        return json.loads(json_text)

    # If no JSON found, try parsing the entire output as fallback
    return json.loads(output.strip())


def test_spaces_list_command():
    """Test spaces list command"""
    result = subprocess.run(
        ["uv", "run", "confluence-gateway", "spaces", "list"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # CLI now outputs JSON format (may have info messages before JSON)
    data = parse_cli_json_output(result.stdout)
    assert "spaces" in data
    assert "pagination" in data


def test_spaces_list_all_command():
    """Test spaces list --all command"""
    result = subprocess.run(
        ["uv", "run", "confluence-gateway", "spaces", "list", "--all"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0


def test_spaces_info_command():
    """Test spaces info command (requires actual space key)"""
    # First get actual space key from spaces list
    list_result = subprocess.run(
        ["uv", "run", "confluence-gateway", "spaces", "list"],
        capture_output=True,
        text=True,
    )

    if list_result.returncode == 0 and list_result.stdout:
        # Extract first space key from JSON output
        data = parse_cli_json_output(list_result.stdout)
        if data["spaces"]:
            space_key = data["spaces"][0]["key"]
            info_result = subprocess.run(
                ["uv", "run", "confluence-gateway", "spaces", "info", space_key],
                capture_output=True,
                text=True,
            )
            assert info_result.returncode == 0
            space_data = parse_cli_json_output(info_result.stdout)
            assert "key" in space_data
            assert "name" in space_data
