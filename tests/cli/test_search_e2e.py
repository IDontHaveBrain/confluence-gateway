import pytest

from tests.utils.cli_helpers import (
    CLITestRunner,
    run_search_cql,
    run_search_hybrid,
    run_search_semantic,
    run_search_text,
)


def test_search_text_command():
    """Test search text command"""
    data = run_search_text("test")

    CLITestRunner.assert_search_response(data, expected_fields=["total"])


def test_search_semantic_command():
    """Test search semantic command"""
    data = run_search_semantic("test query")

    CLITestRunner.assert_search_response(data, expected_fields=["query"])


def test_search_cql_command():
    """Test search cql command"""
    data = run_search_cql("text ~ test")

    CLITestRunner.assert_search_response(data)


def test_search_hybrid_command():
    """Test search hybrid command"""
    data = run_search_hybrid("test")

    CLITestRunner.assert_search_response(data)
