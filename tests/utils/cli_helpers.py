"""CLI test utilities for Confluence Gateway.

This module provides standardized utilities for CLI testing to eliminate
duplication across 47+ subprocess execution patterns.
"""

import json
import subprocess
from typing import Any, Optional

from tests.fixtures.test_utils import parse_cli_json_output


class CLITestRunner:
    """Utility class for running CLI commands and handling common assertion patterns."""

    @staticmethod
    def run_command(
        command_args: list[str],
        check_success: bool = True,
        capture_output: bool = True,
        text: bool = True,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Run a CLI command with standard configuration.

        Args:
            command_args: Command arguments (e.g., ["search", "text", "query"])
            check_success: Whether to assert returncode == 0
            capture_output: Whether to capture stdout/stderr
            text: Whether to return text output
            **kwargs: Additional arguments for subprocess.run

        Returns:
            subprocess.CompletedProcess result
        """
        full_command = ["uv", "run", "confluence-gateway"] + command_args

        result = subprocess.run(
            full_command, capture_output=capture_output, text=text, **kwargs
        )

        if check_success:
            assert result.returncode == 0, (
                f"Command failed with return code {result.returncode}. stderr: {result.stderr}"
            )

        return result

    @staticmethod
    def run_command_json(
        command_args: list[str], check_success: bool = True, **kwargs
    ) -> dict[str, Any]:
        """Run a CLI command and return parsed JSON output.

        Args:
            command_args: Command arguments (e.g., ["search", "text", "query"])
            check_success: Whether to assert returncode == 0
            **kwargs: Additional arguments for subprocess.run

        Returns:
            Parsed JSON dictionary from command output
        """
        result = CLITestRunner.run_command(
            command_args, check_success=check_success, **kwargs
        )

        return parse_cli_json_output(result.stdout)

    @staticmethod
    def assert_search_response(
        data: dict[str, Any], expected_fields: list[str] | None = None
    ) -> None:
        """Assert common search response structure.

        Args:
            data: Parsed JSON response data
            expected_fields: Additional fields to check beyond 'results'
        """
        assert "results" in data, "Search response missing 'results' field"

        if expected_fields:
            for field in expected_fields:
                assert field in data, f"Search response missing '{field}' field"

    @staticmethod
    def assert_spaces_response(data: dict[str, Any]) -> None:
        """Assert common spaces response structure.

        Args:
            data: Parsed JSON response data
        """
        assert "spaces" in data, "Spaces response missing 'spaces' field"
        assert "pagination" in data, "Spaces response missing 'pagination' field"

    @staticmethod
    def assert_space_info_response(data: dict[str, Any]) -> None:
        """Assert space info response structure.

        Args:
            data: Parsed JSON response data
        """
        assert "key" in data, "Space info response missing 'key' field"
        assert "name" in data, "Space info response missing 'name' field"

    @staticmethod
    def assert_generation_response(data: dict[str, Any]) -> None:
        """Assert generation response structure.

        Args:
            data: Parsed JSON response data
        """
        assert "answer" in data or "response" in data, (
            "Generation response missing 'answer' or 'response' field"
        )


class CLICommandBuilder:
    """Builder pattern for constructing CLI commands with options."""

    def __init__(self, base_command: list[str]):
        """Initialize with base command.

        Args:
            base_command: Base command parts (e.g., ["search", "text"])
        """
        self.command = base_command.copy()

    def add_option(self, option: str, value: str | None = None) -> "CLICommandBuilder":
        """Add an option to the command.

        Args:
            option: Option name (e.g., "--verbose", "--space")
            value: Option value if applicable

        Returns:
            Self for method chaining
        """
        self.command.append(option)
        if value is not None:
            self.command.append(value)
        return self

    def add_argument(self, arg: str) -> "CLICommandBuilder":
        """Add a positional argument.

        Args:
            arg: Argument value

        Returns:
            Self for method chaining
        """
        self.command.append(arg)
        return self

    def build(self) -> list[str]:
        """Build the final command list.

        Returns:
            Complete command argument list
        """
        return self.command.copy()

    def run(self, check_success: bool = True, **kwargs) -> subprocess.CompletedProcess:
        """Build and run the command.

        Args:
            check_success: Whether to assert returncode == 0
            **kwargs: Additional arguments for subprocess.run

        Returns:
            subprocess.CompletedProcess result
        """
        return CLITestRunner.run_command(
            self.build(), check_success=check_success, **kwargs
        )

    def run_json(self, check_success: bool = True, **kwargs) -> dict[str, Any]:
        """Build and run the command, returning parsed JSON.

        Args:
            check_success: Whether to assert returncode == 0
            **kwargs: Additional arguments for subprocess.run

        Returns:
            Parsed JSON dictionary from command output
        """
        return CLITestRunner.run_command_json(
            self.build(), check_success=check_success, **kwargs
        )


# Convenience functions for common command patterns
def run_spaces_list(all_spaces: bool = False, **kwargs) -> dict[str, Any]:
    """Run spaces list command.

    Args:
        all_spaces: Whether to include --all flag
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    builder = CLICommandBuilder(["spaces", "list"])
    if all_spaces:
        builder.add_option("--all")

    return builder.run_json(**kwargs)


def run_spaces_info(space_key: str, **kwargs) -> dict[str, Any]:
    """Run spaces info command.

    Args:
        space_key: Space key to get info for
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["spaces", "info", space_key], **kwargs)


def run_search_text(query: str, **kwargs) -> dict[str, Any]:
    """Run text search command.

    Args:
        query: Search query
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["search", "text", query], **kwargs)


def run_search_semantic(query: str, **kwargs) -> dict[str, Any]:
    """Run semantic search command.

    Args:
        query: Search query
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["search", "semantic", query], **kwargs)


def run_search_hybrid(query: str, **kwargs) -> dict[str, Any]:
    """Run hybrid search command.

    Args:
        query: Search query
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["search", "hybrid", query], **kwargs)


def run_search_cql(query: str, **kwargs) -> dict[str, Any]:
    """Run CQL search command.

    Args:
        query: CQL query
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["search", "cql", query], **kwargs)


def run_generate_answer(question: str, **kwargs) -> dict[str, Any]:
    """Run generate answer command.

    Args:
        question: Question to generate answer for
        **kwargs: Additional arguments for subprocess.run

    Returns:
        Parsed JSON response
    """
    return CLITestRunner.run_command_json(["generate", "answer", question], **kwargs)
