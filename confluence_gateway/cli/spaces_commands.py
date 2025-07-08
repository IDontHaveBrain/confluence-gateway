import csv
import io
import time
from collections.abc import Callable
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.cli.common import handle_cli_errors
from confluence_gateway.cli.dependencies import _get_confluence_client
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    ConfluenceGatewayError,
)

app = typer.Typer()
console = Console()

T = TypeVar("T")


def retry_on_network_error(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = 3,
    verbose: bool = False
) -> T:
    """
    Retry a function on network errors with exponential backoff.

    Args:
        func: Function to retry
        operation_name: Human-readable operation name for error messages
        max_retries: Maximum number of retry attempts
        verbose: Whether to show detailed retry information

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except ConfluenceConnectionError as e:
            last_exception = e
            if attempt < max_retries:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if verbose:
                    console.print(
                        f"[yellow]Network error during {operation_name} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay} seconds...[/yellow]"
                    )
                else:
                    console.print(
                        f"[yellow]Network error. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})[/yellow]"
                    )
                time.sleep(delay)
            else:
                # Final attempt failed
                if verbose:
                    console.print(f"[red]All {max_retries + 1} attempts failed for {operation_name}[/red]")
                break
        except (ConfluenceAuthenticationError, ConfluenceAPIError):
            # Don't retry authentication or API errors
            raise
        except Exception as e:
            # Don't retry unexpected errors
            last_exception = e
            break

    # Re-raise the last exception
    if last_exception:
        raise last_exception
    else:
        # Should never happen, but just in case
        raise RuntimeError(f"Retry logic failed unexpectedly for {operation_name}")


def handle_spaces_error(e: Exception, operation: str, verbose: bool = False) -> None:
    """
    Handle and display user-friendly error messages for spaces operations.

    Args:
        e: The exception that occurred
        operation: Description of the operation that failed
        verbose: Whether to show detailed error information
    """
    if isinstance(e, ConfluenceAuthenticationError):
        console.print(f"[red]❌ Authentication failed during {operation}[/red]")
        console.print("[yellow]💡 Check your credentials:[/yellow]")
        console.print("   • Verify your Confluence URL is correct")
        console.print("   • Ensure your username/email is correct")
        console.print("   • Check that your API token is valid and not expired")
        console.print("   • Confirm you have permission to access spaces")
        if verbose:
            console.print(f"[dim]Technical details: {e}[/dim]")

    elif isinstance(e, ConfluenceConnectionError):
        console.print(f"[red]❌ Network connection failed during {operation}[/red]")
        console.print("[yellow]💡 Troubleshooting steps:[/yellow]")
        console.print("   • Check your internet connection")
        console.print("   • Verify the Confluence URL is reachable")
        console.print("   • Check if there's a firewall or proxy blocking the request")
        console.print("   • Try again in a few moments")
        if verbose:
            console.print(f"[dim]Technical details: {e}[/dim]")

    elif isinstance(e, ConfluenceAPIError):
        console.print(f"[red]❌ Confluence API error during {operation}[/red]")
        if hasattr(e, 'status_code') and e.status_code:
            if e.status_code == 403:
                console.print("[yellow]💡 Permission denied:[/yellow]")
                console.print("   • You may not have permission to view spaces")
                console.print("   • Contact your Confluence administrator")
            elif e.status_code == 404:
                console.print("[yellow]💡 Resource not found:[/yellow]")
                console.print("   • The space or API endpoint may not exist")
                console.print("   • Check your Confluence URL and space key")
            elif e.status_code == 429:
                console.print("[yellow]💡 Rate limit exceeded:[/yellow]")
                console.print("   • Too many requests sent to Confluence")
                console.print("   • Wait a few minutes before trying again")
            elif e.status_code >= 500:
                console.print("[yellow]💡 Server error:[/yellow]")
                console.print("   • Confluence server is experiencing issues")
                console.print("   • Try again later or contact your administrator")
            else:
                console.print(f"[yellow]💡 HTTP {e.status_code} error[/yellow]")

        if verbose and hasattr(e, 'error_message') and e.error_message:
            console.print(f"[dim]API error details: {e.error_message}[/dim]")

    elif isinstance(e, ConfluenceGatewayError):
        console.print(f"[red]❌ Application error during {operation}[/red]")
        console.print(f"[yellow]Error: {e}[/yellow]")
        if verbose:
            console.print(f"[dim]Error type: {type(e).__name__}[/dim]")

    else:
        console.print(f"[red]❌ Unexpected error during {operation}[/red]")
        console.print(f"[yellow]Error: {e}[/yellow]")
        console.print("[yellow]💡 This may be a bug. Please report it with the error details.[/yellow]")
        if verbose:
            console.print(f"[dim]Error type: {type(e).__name__}[/dim]")
            import traceback
            console.print(f"[dim]Stack trace: {traceback.format_exc()}[/dim]")


@app.command("list")
@handle_cli_errors
def list_spaces(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or csv",
    ),
    page: int = typer.Option(
        1,
        "--page",
        "-p",
        help="Page number (starts from 1)",
        min=1,
    ),
    page_size: int = typer.Option(
        25,
        "--page-size",
        "-s",
        help="Number of spaces per page",
        min=1,
        max=100,
    ),
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Fetch all spaces (ignore pagination)",
    ),
    type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by space type: personal, global, or all (default: all)",
    ),
    search: str = typer.Option(
        None,
        "--search",
        help="Search spaces by name or key (case-insensitive)",
    ),
    key_prefix: str = typer.Option(
        None,
        "--key-prefix",
        help="Filter spaces by key prefix (case-insensitive)",
    ),
    sort: str = typer.Option(
        None,
        "--sort",
        help="Sort spaces by: name, key, type, or id",
    ),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        "-r",
        help="Reverse sort order",
    ),
    no_truncate: bool = typer.Option(
        False,
        "--no-truncate",
        help="Do not truncate long text in table format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed error messages and retry information",
    ),
) -> None:
    """
    List Confluence spaces with pagination support.

    Displays accessible Confluence spaces with their keys, names, and types.
    By default shows 25 spaces per page. Use --all to fetch all spaces at once.

    Output formats:
    - table: Rich formatted table with auto-adjusted column widths (default)
    - json: JSON format with pagination metadata
    - csv: CSV format with headers

    Examples:
        confluence-gateway spaces list --type global
        confluence-gateway spaces list --search "dev"
        confluence-gateway spaces list --key-prefix TEAM
        confluence-gateway spaces list --sort name
        confluence-gateway spaces list --sort type --reverse
        confluence-gateway spaces list --format csv --all
        confluence-gateway spaces list --no-truncate
    """
    if format not in ["table", "json", "csv"]:
        console.print("[red]Error: format must be 'table', 'json', or 'csv'[/red]")
        raise typer.Exit(1)

    # Validate type parameter
    if type and type.lower() not in ["personal", "global", "all"]:
        console.print("[red]Error: type must be 'personal', 'global', or 'all'[/red]")
        raise typer.Exit(1)

    # Validate sort parameter
    if sort and sort.lower() not in ["name", "key", "type", "id"]:
        console.print("[red]Error: sort must be 'name', 'key', 'type', or 'id'[/red]")
        raise typer.Exit(1)

    try:
        client: ConfluenceClient = _get_confluence_client()

        # Prepare space_type parameter for API
        space_type = None if not type or type.lower() == "all" else type.lower()

        if all:
            # Fetch all spaces at once
            if format not in ["json", "csv"]:
                # Show progress indicator for table format
                with Status(
                    "[cyan]Fetching all spaces...", spinner="dots", console=console
                ) as status:
                    spaces = []

                    # Use retry logic for network errors
                    def fetch_all_spaces() -> Any:
                        return client.list_all_spaces(
                            space_type=space_type,
                            space_status="current",
                        )

                    spaces = retry_on_network_error(
                        fetch_all_spaces,
                        "fetching all spaces",
                        verbose=verbose
                    )

                    status.update(f"[green]✓ Fetched {len(spaces)} spaces")
            else:
                # For JSON and CSV output, fetch without progress indicator
                def fetch_all_spaces() -> Any:
                    return client.list_all_spaces(
                        space_type=space_type,
                        space_status="current",
                    )

                spaces = retry_on_network_error(
                    fetch_all_spaces,
                    "fetching all spaces",
                    verbose=verbose
                )

            # Apply client-side filtering
            if search:
                search_lower = search.lower()
                spaces = [
                    s
                    for s in spaces
                    if search_lower in s.key.lower()
                    or search_lower in (s.name or s.title).lower()
                ]

            if key_prefix:
                prefix_lower = key_prefix.lower()
                spaces = [s for s in spaces if s.key.lower().startswith(prefix_lower)]

            # Apply sorting if requested
            if sort:
                sort_key = sort.lower()
                if sort_key == "name":
                    spaces.sort(
                        key=lambda s: (s.name or s.title).lower(), reverse=reverse
                    )
                elif sort_key == "key":
                    spaces.sort(key=lambda s: s.key.lower(), reverse=reverse)
                elif sort_key == "type":
                    spaces.sort(
                        key=lambda s: s.type.value if s.type else "unknown",
                        reverse=reverse,
                    )
                elif sort_key == "id":
                    spaces.sort(key=lambda s: s.id, reverse=reverse)

            total_count = len(spaces)
            current_page = 1
            total_pages = 1
        else:
            # Use pagination - Note: search and key_prefix filtering happens server-side
            # but we'll need to fetch all and filter for accurate counts
            start = (page - 1) * page_size

            if search or key_prefix or sort:
                # If we have client-side filters or sorting, we need to fetch all spaces first
                if format not in ["json", "csv"]:
                    with Status(
                        "[cyan]Fetching spaces for filtering...",
                        spinner="dots",
                        console=console,
                    ) as status:
                        def fetch_all_spaces_for_filtering() -> Any:
                            return client.list_all_spaces(
                                space_type=space_type, space_status="current"
                            )

                        all_spaces = retry_on_network_error(
                            fetch_all_spaces_for_filtering,
                            "fetching spaces for filtering",
                            verbose=verbose
                        )
                        status.update(
                            f"[green]✓ Fetched {len(all_spaces)} spaces for filtering"
                        )
                else:
                    # For JSON and CSV output, fetch without progress indicator
                    def fetch_all_spaces_for_filtering() -> Any:
                        return client.list_all_spaces(
                            space_type=space_type, space_status="current"
                        )

                    all_spaces = retry_on_network_error(
                        fetch_all_spaces_for_filtering,
                        "fetching spaces for filtering",
                        verbose=verbose
                    )

                # Apply client-side filtering
                if search:
                    search_lower = search.lower()
                    all_spaces = [
                        s
                        for s in all_spaces
                        if search_lower in s.key.lower()
                        or search_lower in (s.name or s.title).lower()
                    ]

                if key_prefix:
                    prefix_lower = key_prefix.lower()
                    all_spaces = [
                        s for s in all_spaces if s.key.lower().startswith(prefix_lower)
                    ]

                # Apply sorting before pagination
                if sort:
                    sort_key = sort.lower()
                    if sort_key == "name":
                        all_spaces.sort(
                            key=lambda s: (s.name or s.title).lower(), reverse=reverse
                        )
                    elif sort_key == "key":
                        all_spaces.sort(key=lambda s: s.key.lower(), reverse=reverse)
                    elif sort_key == "type":
                        all_spaces.sort(
                            key=lambda s: s.type.value if s.type else "unknown",
                            reverse=reverse,
                        )
                    elif sort_key == "id":
                        all_spaces.sort(key=lambda s: s.id, reverse=reverse)

                # Apply pagination to filtered/sorted results
                total_count = len(all_spaces)
                spaces = all_spaces[start : start + page_size]
                current_page = page
                total_pages = (
                    (total_count + page_size - 1) // page_size if total_count > 0 else 1
                )
            else:
                # No client-side filtering or sorting, use server-side pagination
                if format not in ["json", "csv"]:
                    with Status(
                        f"[cyan]Fetching page {page} of spaces...",
                        spinner="dots",
                        console=console,
                    ) as status:
                        def fetch_spaces_paginated() -> Any:
                            return client.list_spaces_paginated(
                                start=start,
                                limit=page_size,
                                space_type=space_type,
                                space_status="current",
                            )

                        spaces, total_count = retry_on_network_error(
                            fetch_spaces_paginated,
                            f"fetching page {page} of spaces",
                            verbose=verbose
                        )
                        status.update(
                            f"[green]✓ Fetched {len(spaces)} spaces from page {page}"
                        )
                else:
                    # For JSON and CSV output, fetch without progress indicator
                    def fetch_spaces_paginated() -> Any:
                        return client.list_spaces_paginated(
                            start=start,
                            limit=page_size,
                            space_type=space_type,
                            space_status="current",
                        )

                    spaces, total_count = retry_on_network_error(
                        fetch_spaces_paginated,
                        f"fetching page {page} of spaces",
                        verbose=verbose
                    )
                current_page = page
                total_pages = (
                    (total_count + page_size - 1) // page_size if total_count > 0 else 1
                )

        if not spaces:
            if page > 1:
                console.print(
                    f"[yellow]No spaces found on page {page}. Try a lower page number.[/yellow]"
                )
            else:
                console.print(
                    "[yellow]No spaces found or no access to any spaces.[/yellow]"
                )
            return

        if format == "json":
            import json

            # Convert spaces to JSON-serializable format
            spaces_data = []
            for space in spaces:
                space_dict = {
                    "id": space.id,
                    "key": space.key,
                    "name": space.name or space.title,
                    "title": space.title,
                    "type": space.type.value if space.type else "unknown",
                }
                if hasattr(space, "description_text") and space.description_text:
                    space_dict["description"] = space.description_text
                spaces_data.append(space_dict)

            result = {
                "spaces": spaces_data,
                "pagination": {
                    "page": current_page,
                    "page_size": page_size if not all else len(spaces),
                    "total_pages": total_pages,
                    "total_count": total_count,
                },
            }
            console.print(json.dumps(result, indent=2))
        elif format == "csv":
            # CSV format output
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            headers = ["Key", "Name", "Type", "ID", "Description"]
            writer.writerow(headers)

            # Write data rows
            for space in spaces:
                row = [
                    space.key,
                    space.name or space.title,
                    space.type.value if space.type else "unknown",
                    space.id,
                    getattr(space, "description_text", "") or "",
                ]
                writer.writerow(row)

            # Print CSV output
            console.print(output.getvalue().rstrip(), highlight=False)

            # Add pagination info as CSV comment if not showing all
            if not all and total_pages > 1:
                console.print(
                    f"\n# Page {current_page} of {total_pages}, {total_count} total spaces",
                    style="dim",
                )
        else:
            # Table format
            if all:
                table = Table(title=f"Confluence Spaces ({total_count} total)")
            else:
                table = Table(
                    title=f"Confluence Spaces (Page {current_page}/{total_pages}, {total_count} total)"
                )

            # Calculate column widths based on content
            key_width = max((len(s.key) for s in spaces), default=3)
            name_width = max((len(s.name or s.title) for s in spaces), default=4)
            type_width = max(
                (len(s.type.value if s.type else "unknown") for s in spaces), default=4
            )
            id_width = max((len(s.id) for s in spaces), default=2)

            # Add some padding
            key_width = max(key_width + 2, 10)
            name_width = max(name_width + 2, 20)
            type_width = max(type_width + 2, 10)
            id_width = max(id_width + 2, 10)

            # If no_truncate is False, apply maximum widths
            if not no_truncate:
                name_width = min(name_width, 50)  # Max 50 chars for name
                id_width = min(id_width, 20)  # Max 20 chars for ID

            table.add_column("Key", style="cyan", no_wrap=True, width=key_width)
            table.add_column(
                "Name",
                style="green",
                width=name_width if not no_truncate else None,
                overflow="ellipsis" if not no_truncate else "fold",
            )
            table.add_column("Type", style="magenta", width=type_width)
            table.add_column(
                "ID",
                style="dim",
                width=id_width if not no_truncate else None,
                overflow="ellipsis" if not no_truncate else "fold",
            )

            for space in spaces:
                space_type = space.type.value if space.type else "unknown"
                space_name = space.name or space.title
                table.add_row(space.key, space_name, space_type, space.id)

            console.print(table)

            if not all and total_pages > 1:
                console.print(
                    f"\n[dim]Showing page {current_page} of {total_pages}. "
                    f"Use --page to navigate or --all to show all spaces.[/dim]"
                )

    except Exception as e:
        handle_spaces_error(e, "listing spaces", verbose=verbose)
        raise typer.Exit(1)


@app.command("info")
@handle_cli_errors
def space_info(
    space_key: str = typer.Argument(help="The space key to get information about"),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed error messages and retry information",
    ),
) -> None:
    """
    Get detailed information about a specific Confluence space.
    """
    try:
        client: ConfluenceClient = _get_confluence_client()

        with Status(
            f"[cyan]Fetching information for space '{space_key}'...",
            spinner="dots",
            console=console,
        ) as status:
            def fetch_space_info() -> Any:
                return client.get_space(space_key)

            space = retry_on_network_error(
                fetch_space_info,
                f"fetching information for space '{space_key}'",
                verbose=verbose
            )
            status.update(f"[green]✓ Retrieved information for space '{space_key}'")

        console.print("\n[bold cyan]Space Information[/bold cyan]")
        console.print(f"[bold]Key:[/bold] {space.key}")
        console.print(f"[bold]Name:[/bold] {space.name or space.title}")
        console.print(f"[bold]ID:[/bold] {space.id}")
        console.print(
            f"[bold]Type:[/bold] {space.type.value if space.type else 'unknown'}"
        )

        if hasattr(space, "description_text") and space.description_text:
            console.print(f"[bold]Description:[/bold] {space.description_text}")

        if space.created_at:
            console.print(f"[bold]Created:[/bold] {space.created_at}")

        if space.updated_at:
            console.print(f"[bold]Updated:[/bold] {space.updated_at}")

    except Exception as e:
        handle_spaces_error(e, f"getting information for space '{space_key}'", verbose=verbose)
        raise typer.Exit(1)
