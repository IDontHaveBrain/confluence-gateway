import logging
from functools import wraps

import typer

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.api.schemas.responses import (
    IndexingStatusResponse,
    SearchResultItem,
    SourceDocument,
)
from confluence_gateway.core.exceptions import ConfluenceGatewayError

logger = logging.getLogger(__name__)


# Utility functions for text formatting
def create_table(
    title: str, columns: list[tuple[str, int]], rows: list[list[str]]
) -> str:
    """Create a simple text-based table."""
    lines = []

    # Title
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))

    # Calculate column widths
    col_names = [col[0] for col in columns]
    col_widths = [col[1] for col in columns]

    # Adjust widths based on content if needed
    for i, name in enumerate(col_names):
        col_widths[i] = max(col_widths[i], len(name))

    # Header
    header = " | ".join(name.ljust(width) for name, width in zip(col_names, col_widths))
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for row in rows:
        formatted_row = []
        for i, cell in enumerate(row):
            # Truncate if needed
            if len(cell) > col_widths[i]:
                cell = cell[: col_widths[i] - 3] + "..."
            formatted_row.append(cell.ljust(col_widths[i]))
        lines.append(" | ".join(formatted_row))

    return "\n".join(lines)


def create_panel(content: str, title: str | None = None) -> str:
    """Create a simple bordered panel."""
    lines = content.split("\n")
    max_width = max(len(line) for line in lines) if lines else 0

    if title:
        max_width = max(max_width, len(title) + 4)

    result = []

    # Top border
    if title:
        padding = max_width - len(title) - 2
        left_pad = padding // 2
        right_pad = padding - left_pad
        result.append("┌" + "─" * left_pad + f" {title} " + "─" * right_pad + "┐")
    else:
        result.append("┌" + "─" * (max_width + 2) + "┐")

    # Content
    for line in lines:
        padding = max_width - len(line)
        result.append(f"│ {line}{' ' * padding} │")

    # Bottom border
    result.append("└" + "─" * (max_width + 2) + "┘")

    return "\n".join(result)


def print_status(message: str, status_type: str = "info"):
    """Print a status message with optional prefix."""
    prefix_map = {
        "info": "[INFO]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "success": "[OK]",
        "dim": "",
    }
    prefix = prefix_map.get(status_type, "")
    if prefix:
        print(f"{prefix} {message}")
    else:
        print(message)


def print_search_results(
    results: list[SearchResultItem], total: int, start: int, limit: int, took_ms: float
):
    if not results:
        print("No results found.")
        return

    # Prepare table data
    columns = [
        ("ID", 12),
        ("Title", 30),
        ("Type", 10),
        ("Space", 20),
        ("Last Modified", 20),
        ("URL", 40),
    ]

    rows = []
    for item in results:
        last_modified_str = (
            item.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            if item.last_modified
            else "N/A"
        )
        space_display = f"{item.space_name or ''} ({item.space_key or ''})"

        rows.append(
            [
                item.id,
                item.title,
                item.type,
                space_display,
                last_modified_str,
                item.url or "N/A",
            ]
        )

    # Print table
    table_str = create_table("Search Results", columns, rows)
    print(table_str)

    # Print summary
    start_num = start + 1
    end_num = start + len(results)
    print(f"\nShowing results {start_num}-{end_num} of {total}. Took {took_ms:.2f} ms.")


def print_semantic_search_results(
    results: list[VectorSearchResultItem], query: str, took_ms: float
):
    if not results:
        print("No semantic results found.")
        return

    # Prepare table data
    columns = [
        ("ID", 20),
        ("Score", 8),
        ("Title", 30),
        ("Space", 15),
        ("URL", 40),
        ("Text Snippet", 60),
    ]

    rows = []
    for item in results:
        metadata = item.metadata or {}
        title = metadata.get("title", "N/A")
        space_key = metadata.get("space_key", "N/A")
        url = metadata.get("url", "N/A")
        snippet = (item.text or "").replace("\n", " ").strip()
        snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet

        rows.append([item.id, f"{item.score:.3f}", title, space_key, url, snippet])

    # Print table
    table_str = create_table(f"Semantic Search Results for: '{query}'", columns, rows)
    print(table_str)

    # Print summary
    print(f"\nSemantic search returned {len(results)} results. Took {took_ms:.2f} ms.")


def print_indexing_status(status: IndexingStatusResponse):
    status_type = {
        "idle": "success",
        "success": "success",
        "running": "warning",
        "failure": "error",
    }.get(status.status, "info")

    print_status(f"Indexing Status: {status.status}", status_type)

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

    print(f"  Last Run Start: {start_time}")
    print(f"  Last Run End:   {end_time}")

    if status.last_error_message:
        print_status(f"  Last Error: {status.last_error_message}", "error")


def print_generated_answer(answer: str, sources: list[SourceDocument]):
    # Print answer in a panel
    panel_str = create_panel(answer, "Generated Answer")
    print(panel_str)

    if not sources:
        print_status("No sources provided for this answer.", "dim")
        return

    # Prepare sources table
    columns = [("ID", 20), ("Score", 8), ("Title", 30), ("Space", 15), ("URL", 40)]

    rows = []
    for source in sources:
        rows.append(
            [
                source.id,
                f"{source.score:.3f}",
                source.title or "N/A",
                source.space_key or "N/A",
                source.url or "N/A",
            ]
        )

    # Print sources table
    table_str = create_table("Sources", columns, rows)
    print(f"\n{table_str}")


def handle_cli_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfluenceGatewayError as e:
            error_type = type(e).__name__
            logger.error(f"CLI Error ({error_type}): {e}", exc_info=True)
            print_status(f"Error ({error_type}): {str(e)}", "error")
            raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except Exception as e:
            logger.error(f"Unexpected CLI Error: {e}", exc_info=True)
            print_status(f"Unexpected Error: {str(e)}", "error")
            print_status("Check logs for more details.", "dim")
            raise typer.Exit(code=1)

    return wrapper
