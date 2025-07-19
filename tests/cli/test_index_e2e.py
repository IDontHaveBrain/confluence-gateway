import pytest

from tests.utils.cli_helpers import CLITestRunner


def test_index_trigger_command():
    """Test index trigger command"""
    # Indexing can take a long time, so just check successful start
    CLITestRunner.run_command(["index", "trigger"])

    # Success is verified by CLITestRunner.run_command's check_success=True (default)


def test_index_status_command():
    """Test index status command"""
    data = CLITestRunner.run_command_json(["index", "status"])

    assert "status" in data
