"""Shared test utilities for Confluence Gateway tests.

This module contains common helper functions used across multiple test files
to reduce code duplication and ensure consistency.
"""

import json
from typing import Any
from unittest.mock import MagicMock


def parse_cli_json_output(output: str) -> dict[str, Any]:
    """Parse JSON from CLI output that may contain info messages before JSON.

    Args:
        output: Raw CLI output string

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If no valid JSON found in output
    """
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


def create_mock_litellm_response(content: str) -> MagicMock:
    """Create a mock LiteLLM response object.

    Args:
        content: Response content to return

    Returns:
        Mock response object matching LiteLLM structure
    """
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    return mock_response
