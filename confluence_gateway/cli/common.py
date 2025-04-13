import logging
from functools import wraps

import typer
from rich import print as rich_print
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.api.schemas.responses import (
    IndexingStatusResponse,
    SearchResultItem,
    SourceDocument,
)
from confluence_gateway.core.exceptions import ConfluenceGatewayError

logger = logging.getLogger(__name__)
console = Console()


def print_search_results(
    results: list[SearchResultItem], total: int, start: int, limit: int, took_ms: float
):
    table = Table(
        title="Search Results",
        show_header=True,
        header_style="bold magenta",
        box=None,
    )
    table.add_column("ID", style="dim", width=12)
    table.add_column("Title", style="bold")
    table.add_column("Type", width=10)
    table.add_column("Space", width=15)
    table.add_column("Last Modified", width=20)
    table.add_column("URL")

    if not results:
        console.print("No results found.")
        return

    for item in results:
        last_modified_str = (
            item.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            if item.last_modified
            else "N/A"
        )
        space_display = (
            f"{escape(item.space_name or '')} ({escape(item.space_key or '')})"
        )
        table.add_row(
            escape(item.id),
            escape(item.title),
            escape(item.type),
            space_display,
            last_modified_str,
            escape(item.url or "N/A"),
        )

    console.print(table)

    start_num = start + 1
    end_num = start + len(results)
    console.print(
        f"Showing results {start_num}-{end_num} of {total}. Took {took_ms:.2f} ms."
    )


def print_semantic_search_results(
    results: list[VectorSearchResultItem], query: str, took_ms: float
):
    table = Table(
        title=f"Semantic Search Results for: '{escape(query)}'",
        show_header=True,
        header_style="bold magenta",
        box=None,
    )
    table.add_column("ID", style="dim", width=20)
    table.add_column("Score", width=8)
    table.add_column("Title", style="bold")
    table.add_column("Space", width=15)
    table.add_column("URL")
    table.add_column("Text Snippet", max_width=60)

    if not results:
        console.print("No semantic results found.")
        return

    for item in results:
        metadata = item.metadata or {}
        title = metadata.get("title", "N/A")
        space_key = metadata.get("space_key", "N/A")
        url = metadata.get("url", "N/A")
        snippet = (item.text or "").replace("\n", " ").strip()
        snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet

        table.add_row(
            escape(item.id),
            f"{item.score:.3f}",
            escape(title),
            escape(space_key),
            escape(url),
            escape(snippet),
        )

    console.print(table)
    console.print(
        f"Semantic search returned {len(results)} results. Took {took_ms:.2f} ms."
    )


def print_indexing_status(status: IndexingStatusResponse):
    status_color = {
        "idle": "green",
        "success": "green",
        "running": "yellow",
        "failure": "red",
    }.get(status.status, "white")

    rich_print(f"[bold]Indexing Status:[/bold] [{status_color}]{status.status}[/]")

    start_time = (
        status.last_run_start_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        if status.last_run_start_time
        else "N/A"
    )
    end_time = (
        status.last_run_end_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        if status.last_run_end_time
        else "N/A"
    )

    rich_print(f"  Last Run Start: {start_time}")
    rich_print(f"  Last Run End:   {end_time}")

    if status.last_error_message:
        rich_print(
            f"  [bold red]Last Error:[/bold red] {escape(status.last_error_message)}"
        )


def print_generated_answer(answer: str, sources: list[SourceDocument]):
    rich_print(
        Panel(
            escape(answer),
            title="[bold cyan]Generated Answer[/bold cyan]",
            border_style="cyan",
        )
    )

    if not sources:
        rich_print("[dim]No sources provided for this answer.[/dim]")
        return

    table = Table(
        title="Sources", show_header=True, header_style="bold magenta", box=None
    )
    table.add_column("ID", style="dim", width=20)
    table.add_column("Score", width=8)
    table.add_column("Title", style="bold")
    table.add_column("Space", width=15)
    table.add_column("URL")

    for source in sources:
        table.add_row(
            escape(source.id),
            f"{source.score:.3f}",
            escape(source.title or "N/A"),
            escape(source.space_key or "N/A"),
            escape(source.url or "N/A"),
        )

    console.print(table)


def handle_cli_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfluenceGatewayError as e:
            error_type = type(e).__name__
            logger.error(f"CLI Error ({error_type}): {e}", exc_info=True)
            rich_print(f"[bold red]Error ({error_type}):[/bold red] {escape(str(e))}")
            raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except Exception as e:
            logger.error(f"Unexpected CLI Error: {e}", exc_info=True)
            rich_print(f"[bold red]Unexpected Error:[/bold red] {escape(str(e))}")
            rich_print("[dim]Check logs for more details.[/dim]")
            raise typer.Exit(code=1)

    return wrapper
