import pytest

from tests.utils.cli_helpers import CLITestRunner, run_generate_answer


def test_generate_answer_command():
    """Test generate answer command"""
    data = run_generate_answer("What is Confluence?")

    CLITestRunner.assert_generation_response(data)
    assert "sources" in data
