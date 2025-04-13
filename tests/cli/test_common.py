from datetime import datetime, timezone
from typing import Optional, Literal
from unittest.mock import patch

import pytest
import typer
from rich.panel import Panel
from rich.table import Table

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.api.schemas.responses import (
    IndexingStatusResponse,
    SearchResultItem,
    SourceDocument,
)
from confluence_gateway.cli.common import (
    handle_cli_errors,
    print_generated_answer,
    print_indexing_status,
    print_search_results,
    print_semantic_search_results,
)
from confluence_gateway.core.exceptions import (
    SearchParameterError,
)


def create_sample_search_result_item(i: int) -> SearchResultItem:
    now = datetime.now(timezone.utc)
    return SearchResultItem(
        id=f"page_{i}",
        title=f"Test Page {i}",
        type="page",
        space_key=f"SPC{i}",
        space_name=f"Space {i}",
        url=f"http://confluence.test/display/SPC{i}/Test+Page+{i}",
        excerpt=f"Excerpt for page {i}...",
        last_modified=now,
    )

def create_sample_vector_search_item(i: int) -> VectorSearchResultItem:
    return VectorSearchResultItem(
        id=f"chunk_{i}",
        score=0.85 + (i * 0.01),
        metadata={
            "title": f"Vector Doc {i}",
            "space_key": f"VEC{i}",
            "url": f"http://confluence.test/display/VEC{i}/Vector+Doc+{i}",
        },
        text=f"This is the text snippet for vector document {i}. It contains relevant information.",
    )

def create_sample_indexing_status(
    status: Literal["idle", "running", "success", "failure"], error: Optional[str] = None
) -> IndexingStatusResponse:
    now = datetime.now(timezone.utc)
    return IndexingStatusResponse(
        status=status,
        last_run_start_time=now,
        last_run_end_time=now if status != "running" else None,
        last_error_message=error,
    )

def create_sample_source_document(i: int) -> SourceDocument:
    return SourceDocument(
        id=f"src_chunk_{i}",
        score=0.90 - (i * 0.02),
        title=f"Source Doc {i}",
        url=f"http://confluence.test/display/SRC{i}/Source+Doc+{i}",
        space_key=f"SRC{i}",
    )

class TestPrintFunctions:
    @patch("confluence_gateway.cli.common.console")
    def test_print_search_results_with_data(self, mock_console):
        sample_results = [create_sample_search_result_item(i) for i in range(2)]
        total, start, limit, took_ms = 10, 0, 5, 123.45

        print_search_results(sample_results, total, start, limit, took_ms)

        assert mock_console.print.call_count == 2
        args_table, _ = mock_console.print.call_args_list[0]
        assert isinstance(args_table[0], Table)
        assert "Title" in [col.header for col in args_table[0].columns]
        assert "Last Modified" in [col.header for col in args_table[0].columns]
        args_summary, _ = mock_console.print.call_args_list[1]
        assert isinstance(args_summary[0], str)
        assert f"Showing results {start + 1}-{start + len(sample_results)} of {total}" in args_summary[0]
        assert f"{took_ms:.2f} ms" in args_summary[0]

    @patch("confluence_gateway.cli.common.console")
    def test_print_search_results_no_data(self, mock_console):
        print_search_results([], 0, 0, 5, 10.0)
        mock_console.print.assert_called_once_with("No results found.")

    @patch("confluence_gateway.cli.common.console")
    def test_print_semantic_search_results_with_data(self, mock_console):
        sample_results = [create_sample_vector_search_item(i) for i in range(3)]
        query, took_ms = "semantic query", 55.5

        print_semantic_search_results(sample_results, query, took_ms)

        assert mock_console.print.call_count == 2
        args_table, _ = mock_console.print.call_args_list[0]
        assert isinstance(args_table[0], Table)
        assert "Score" in [col.header for col in args_table[0].columns]
        assert "Text Snippet" in [col.header for col in args_table[0].columns]
        assert f"'{query}'" in args_table[0].title
        args_summary, _ = mock_console.print.call_args_list[1]
        assert isinstance(args_summary[0], str)
        assert f"returned {len(sample_results)} results" in args_summary[0]
        assert f"{took_ms:.2f} ms" in args_summary[0]

    @patch("confluence_gateway.cli.common.console")
    def test_print_semantic_search_results_no_data(self, mock_console):
        print_semantic_search_results([], "empty query", 10.0)
        mock_console.print.assert_called_once_with("No semantic results found.")

    @patch("confluence_gateway.cli.common.rich_print")
    def test_print_indexing_status_success(self, mock_rich_print):
        status_data = create_sample_indexing_status("success")

        print_indexing_status(status_data)

        call_texts = " ".join([c[0][0] for c in mock_rich_print.call_args_list])
        assert "[bold]Indexing Status:[/bold]" in call_texts
        assert "[green]success[/]" in call_texts
        assert "Last Run Start:" in call_texts
        assert "Last Run End:" in call_texts
        assert "Last Error:" not in call_texts

    @patch("confluence_gateway.cli.common.rich_print")
    def test_print_indexing_status_failure(self, mock_rich_print):
        error_msg = "Something went wrong during indexing"
        status_data = create_sample_indexing_status("failure", error=error_msg)

        print_indexing_status(status_data)

        call_texts = " ".join([c[0][0] for c in mock_rich_print.call_args_list])
        assert "[bold]Indexing Status:[/bold]" in call_texts
        assert "[red]failure[/]" in call_texts
        assert "[bold red]Last Error:[/bold red]" in call_texts
        assert error_msg in call_texts

    @patch("confluence_gateway.cli.common.console")
    @patch("confluence_gateway.cli.common.rich_print")
    def test_print_generated_answer_with_sources(self, mock_rich_print, mock_console):
        answer = "This is the generated answer."
        sources = [create_sample_source_document(i) for i in range(2)]

        print_generated_answer(answer, sources)

        panel_call = next((call for call in mock_rich_print.call_args_list if isinstance(call[0][0], Panel)), None)
        assert panel_call is not None
        assert answer in str(panel_call[0][0].renderable)
        assert "[bold cyan]Generated Answer[/bold cyan]" in panel_call[0][0].title

        table_call = next((call for call in mock_console.print.call_args_list if isinstance(call[0][0], Table)), None)
        assert table_call is not None
        assert "Sources" in table_call[0][0].title
        assert "Score" in [col.header for col in table_call[0][0].columns]
        assert "URL" in [col.header for col in table_call[0][0].columns]

    @patch("confluence_gateway.cli.common.console")
    @patch("confluence_gateway.cli.common.rich_print")
    def test_print_generated_answer_no_sources(self, mock_rich_print, mock_console):
        answer = "Answer without sources."

        print_generated_answer(answer, [])

        assert any(isinstance(call[0][0], Panel) for call in mock_rich_print.call_args_list)
        # Check that rich_print was called at least twice (Panel + dim string)
        assert len(mock_rich_print.call_args_list) >= 2

        # Get the arguments of the second call
        second_call_args = mock_rich_print.call_args_list[1][0] # Positional args

        # Assert the first argument of the second call is the expected string
        assert isinstance(second_call_args[0], str)
        assert "[dim]No sources provided for this answer.[/dim]" in second_call_args[0]
        assert not any(isinstance(call[0][0], Table) for call in mock_console.print.call_args_list)

class TestHandleCliErrors:
    def test_success_case(self):
        @handle_cli_errors
        def successful_func():
            return "Success"

        result = successful_func()

        assert result == "Success"

    @patch("confluence_gateway.cli.common.logger")
    @patch("confluence_gateway.cli.common.rich_print")
    def test_handles_confluence_gateway_error(self, mock_rich_print, mock_logger):
        error_message = "Specific gateway error occurred"
        error_instance = SearchParameterError(error_message)

        @handle_cli_errors
        def gateway_error_func():
            raise error_instance

        with pytest.raises(typer.Exit) as exc_info:
            gateway_error_func()

        assert exc_info.value.exit_code == 1
        mock_rich_print.assert_called_once()
        print_call_args = mock_rich_print.call_args[0][0]
        assert "[bold red]Error (SearchParameterError):[/bold red]" in print_call_args
        assert error_message in print_call_args
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args
        assert "CLI Error (SearchParameterError)" in log_call_args[0][0]
        assert error_message in log_call_args[0][0]
        assert log_call_args[1]["exc_info"] is True

    @patch("confluence_gateway.cli.common.logger")
    @patch("confluence_gateway.cli.common.rich_print")
    def test_handles_unexpected_error(self, mock_rich_print, mock_logger):
        error_message = "Something unexpected happened"
        error_instance = ValueError(error_message)

        @handle_cli_errors
        def unexpected_error_func():
            raise error_instance

        with pytest.raises(typer.Exit) as exc_info:
            unexpected_error_func()

        assert exc_info.value.exit_code == 1
        assert mock_rich_print.call_count == 2
        calls = mock_rich_print.call_args_list
        assert "[bold red]Unexpected Error:[/bold red]" in calls[0][0][0]
        assert error_message in calls[0][0][0]
        assert "[dim]Check logs for more details.[/dim]" in calls[1][0][0]
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args
        assert "Unexpected CLI Error" in log_call_args[0][0]
        assert error_message in log_call_args[0][0]
        assert log_call_args[1]["exc_info"] is True

    def test_allows_typer_exit_to_propagate(self):
        expected_exit_code = 5

        @handle_cli_errors
        def typer_exit_func():
            print(f"Raising typer exit with code {expected_exit_code}")
            raise typer.Exit(code=expected_exit_code)

        with pytest.raises(typer.Exit) as exc_info:
            typer_exit_func()

        assert exc_info.value.exit_code == expected_exit_code
