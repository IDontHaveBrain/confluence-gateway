import pytest

from tests.utils.cli_helpers import (
    CLITestRunner,
    run_spaces_info,
    run_spaces_list,
)


def test_spaces_list_command():
    """Test spaces list command"""
    data = run_spaces_list()

    CLITestRunner.assert_spaces_response(data)


def test_spaces_list_all_command():
    """Test spaces list --all command"""
    data = run_spaces_list(all_spaces=True)

    CLITestRunner.assert_spaces_response(data)


def test_spaces_info_command():
    """Test spaces info command (requires actual space key)"""
    # First get actual space key from spaces list
    data = run_spaces_list()

    if data["spaces"]:
        space_key = data["spaces"][0]["key"]
        space_data = run_spaces_info(space_key)

        CLITestRunner.assert_space_info_response(space_data)
