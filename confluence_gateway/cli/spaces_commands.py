import json
import time
from collections.abc import Callable
from typing import Any, TypeVar

import typer

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

T = TypeVar("T")


def retry_on_network_error(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = 3,
    verbose: bool = False,
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
                delay = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                if verbose:
                    print(
                        f"Network error during {operation_name} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay} seconds..."
                    )
                else:
                    print(
                        f"Network error. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                time.sleep(delay)
            else:
                # Final attempt failed
                if verbose:
                    print(f"All {max_retries + 1} attempts failed for {operation_name}")
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
        print(f"❌ Authentication failed during {operation}")
        print("💡 Check your credentials:")
        print("   • Verify your Confluence URL is correct")
        print("   • Ensure your username/email is correct")
        print("   • Check that your API token is valid and not expired")
        print("   • Confirm you have permission to access spaces")
        if verbose:
            print(f"Technical details: {e}")

    elif isinstance(e, ConfluenceConnectionError):
        print(f"❌ Network connection failed during {operation}")
        print("💡 Troubleshooting steps:")
        print("   • Check your internet connection")
        print("   • Verify the Confluence URL is reachable")
        print("   • Check if there's a firewall or proxy blocking the request")
        print("   • Try again in a few moments")
        if verbose:
            print(f"Technical details: {e}")

    elif isinstance(e, ConfluenceAPIError):
        print(f"❌ Confluence API error during {operation}")
        if hasattr(e, "status_code") and e.status_code:
            if e.status_code == 403:
                print("💡 Permission denied:")
                print("   • You may not have permission to view spaces")
                print("   • Contact your Confluence administrator")
            elif e.status_code == 404:
                print("💡 Resource not found:")
                print("   • The space or API endpoint may not exist")
                print("   • Check your Confluence URL and space key")
            elif e.status_code == 429:
                print("💡 Rate limit exceeded:")
                print("   • Too many requests sent to Confluence")
                print("   • Wait a few minutes before trying again")
            elif e.status_code >= 500:
                print("💡 Server error:")
                print("   • Confluence server is experiencing issues")
                print("   • Try again later or contact your administrator")
            else:
                print(f"💡 HTTP {e.status_code} error")

        if verbose and hasattr(e, "error_message") and e.error_message:
            print(f"API error details: {e.error_message}")

    elif isinstance(e, ConfluenceGatewayError):
        print(f"❌ Application error during {operation}")
        print(f"Error: {e}")
        if verbose:
            print(f"Error type: {type(e).__name__}")

    else:
        print(f"❌ Unexpected error during {operation}")
        print(f"Error: {e}")
        print("💡 This may be a bug. Please report it with the error details.")
        if verbose:
            print(f"Error type: {type(e).__name__}")
            import traceback

            print(f"Stack trace: {traceback.format_exc()}")


@app.command("list")
@handle_cli_errors
def list_spaces(
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed error messages and retry information",
    ),
) -> None:
    """
    List Confluence spaces with pagination support.

    Displays accessible Confluence spaces with their keys, names, and types in JSON format.
    By default shows 25 spaces per page. Use --all to fetch all spaces at once.

    Examples:
        confluence-gateway spaces list --type global
        confluence-gateway spaces list --search "dev"
        confluence-gateway spaces list --key-prefix TEAM
        confluence-gateway spaces list --sort name
        confluence-gateway spaces list --sort type --reverse
        confluence-gateway spaces list --all
    """

    # Validate type parameter
    if type and type.lower() not in ["personal", "global", "all"]:
        print("Error: type must be 'personal', 'global', or 'all'")
        raise typer.Exit(1)

    # Validate sort parameter
    if sort and sort.lower() not in ["name", "key", "type", "id"]:
        print("Error: sort must be 'name', 'key', 'type', or 'id'")
        raise typer.Exit(1)

    try:
        client: ConfluenceClient = _get_confluence_client()

        # Prepare space_type parameter for API
        space_type = None if not type or type.lower() == "all" else type.lower()

        if all:
            # Fetch all spaces at once
            def fetch_all_spaces() -> Any:
                return client.list_all_spaces(
                    space_type=space_type,
                    space_status="current",
                )

            spaces = retry_on_network_error(
                fetch_all_spaces, "fetching all spaces", verbose=verbose
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
                def fetch_all_spaces_for_filtering() -> Any:
                    return client.list_all_spaces(
                        space_type=space_type, space_status="current"
                    )

                all_spaces = retry_on_network_error(
                    fetch_all_spaces_for_filtering,
                    "fetching spaces for filtering",
                    verbose=verbose,
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
                    verbose=verbose,
                )
                current_page = page
                total_pages = (
                    (total_count + page_size - 1) // page_size if total_count > 0 else 1
                )

        if not spaces:
            if page > 1:
                result = {
                    "error": f"No spaces found on page {page}. Try a lower page number.",
                    "spaces": [],
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_pages": 0,
                        "total_count": 0,
                    },
                }
            else:
                result = {
                    "error": "No spaces found or no access to any spaces.",
                    "spaces": [],
                    "pagination": {
                        "page": 1,
                        "page_size": page_size,
                        "total_pages": 0,
                        "total_count": 0,
                    },
                }
            print(json.dumps(result, indent=2))
            return

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
        print(json.dumps(result, indent=2))

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

        def fetch_space_info() -> Any:
            return client.get_space(space_key)

        space = retry_on_network_error(
            fetch_space_info,
            f"fetching information for space '{space_key}'",
            verbose=verbose,
        )

        # Convert space to JSON-serializable format
        space_dict = {
            "id": space.id,
            "key": space.key,
            "name": space.name or space.title,
            "title": space.title,
            "type": space.type.value if space.type else "unknown",
        }

        if hasattr(space, "description_text") and space.description_text:
            space_dict["description"] = space.description_text

        if space.created_at:
            space_dict["created_at"] = str(space.created_at)

        if space.updated_at:
            space_dict["updated_at"] = str(space.updated_at)

        print(json.dumps(space_dict, indent=2))

    except Exception as e:
        handle_spaces_error(
            e, f"getting information for space '{space_key}'", verbose=verbose
        )
        raise typer.Exit(1)
