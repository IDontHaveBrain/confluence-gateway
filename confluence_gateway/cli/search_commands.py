import json
import logging
from datetime import datetime
from typing import Any, cast

import typer

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import ConfluencePage
from confluence_gateway.api.schemas.responses import SearchResultItem
from confluence_gateway.cli.common import (
    handle_cli_errors,
    print_search_results,
    print_semantic_search_results,
    print_status,
)
from confluence_gateway.cli.dependencies import _get_search_service
from confluence_gateway.core.exceptions import SearchParameterError
from confluence_gateway.services.search import EnhancedSearchResult, SearchService

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Search Confluence content using various methods.",
)


def _convert_to_search_result_items(
    pages: list[ConfluencePage], client: ConfluenceClient
) -> list[SearchResultItem]:
    items = []
    for page in pages:
        extracted = client.extract_content_fields(page)
        space_key = extracted.get("space_key", "")
        page_id = extracted.get("id", "")
        base_url = client.base_url

        url = extracted.get("url")
        if not url and base_url and space_key and page_id:
            url = f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"
        elif not url:
            url = "URL not available"

        items.append(
            SearchResultItem(
                id=page_id,
                title=extracted.get("title", "Title not available"),
                type=extracted.get("type", "page"),
                space_key=space_key,
                space_name=extracted.get("space_name", "Space name not available"),
                url=url,
                excerpt=getattr(page, "excerpt", None),
                last_modified=extracted.get("updated_at")
                or extracted.get("created_at")
                or datetime.now(),
            )
        )
    return items


@app.command("text", help="Search using text query (standard, advanced, or hybrid).")
@handle_cli_errors
def text_search(
    query: str = typer.Argument(..., help="Text to search for (min 2 chars)."),
    space_key: str | None = typer.Option(
        None, "--space", "-s", help="Filter by space key."
    ),
    content_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by content type (page, blogpost, attachment, comment).",
    ),
    include_archived: bool = typer.Option(
        False, "--archived", help="Include archived content."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Maximum number of results."
    ),
    start: int | None = typer.Option(
        0, "--start", help="Starting position for pagination."
    ),
    expand: list[str] | None = typer.Option(
        None, "--expand", help="Fields to expand (repeatable)."
    ),
    use_hybrid: bool = typer.Option(
        False, "--hybrid", help="Enable hybrid search (keyword + semantic + RRF)."
    ),
    sort_by: list[str] | None = typer.Option(
        None,
        "--sort-by",
        help="Field(s) to sort by. Valid fields: title, created_at, updated_at, score, space_key. Repeatable.",
    ),
    sort_direction: list[str] | None = typer.Option(
        None,
        "--sort-dir",
        help="Sort direction(s) (asc, desc). Must match --sort-by count.",
    ),
    min_relevance: float | None = typer.Option(
        None, "--min-relevance", help="Minimum relevance score (0.0-1.0)."
    ),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Return only the top N results after fetching."
    ),
) -> None:
    search_service: SearchService = _get_search_service()

    enhanced_result: EnhancedSearchResult

    if use_hybrid:
        print_status("Performing Hybrid Search...", "info")
        enhanced_result = cast(
            EnhancedSearchResult,
            search_service.search_hybrid(
                text=query,
                content_type=content_type,
                space_key=space_key,
                include_archived=include_archived,
                limit=limit,
                start=start,
                expand=expand,
                return_enhanced_result=True,
            ),
        )
    else:
        print_status("Performing Text Search...", "info")
        enhanced_result = cast(
            EnhancedSearchResult,
            search_service.search_by_text(
                text=query,
                content_type=content_type,
                space_key=space_key,
                include_archived=include_archived,
                limit=limit,
                start=start,
                expand=expand,
                min_relevance=min_relevance,
                top_n=top_n,
                sort_by=sort_by,
                sort_direction=sort_direction,
                return_enhanced_result=True,
            ),
        )

    search_items = _convert_to_search_result_items(
        enhanced_result.results.results, search_service.client
    )

    print_search_results(
        results=search_items,
        total=enhanced_result.statistics.total_results,
        start=enhanced_result.results.start,
        limit=enhanced_result.results.limit,
        took_ms=enhanced_result.statistics.execution_time_ms,
    )


@app.command("cql", help="Search using a Confluence Query Language (CQL) query.")
@handle_cli_errors
def cql_search(
    cql_query: str = typer.Argument(..., help="The CQL query string."),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Maximum number of results."
    ),
    start: int | None = typer.Option(
        0, "--start", help="Starting position for pagination."
    ),
    expand: list[str] | None = typer.Option(
        None, "--expand", help="Fields to expand (repeatable)."
    ),
) -> None:
    search_service: SearchService = _get_search_service()
    print_status(f"Performing CQL Search: '{cql_query}'", "info")

    enhanced_result: EnhancedSearchResult = cast(
        EnhancedSearchResult,
        search_service.search_by_cql(
            cql=cql_query,
            limit=limit,
            start=start,
            expand=expand,
            return_enhanced_result=True,
        ),
    )

    search_items = _convert_to_search_result_items(
        enhanced_result.results.results, search_service.client
    )

    print_search_results(
        results=search_items,
        total=enhanced_result.statistics.total_results,
        start=enhanced_result.results.start,
        limit=enhanced_result.results.limit,
        took_ms=enhanced_result.statistics.execution_time_ms,
    )


@app.command("semantic", help="Perform semantic search using vector embeddings.")
@handle_cli_errors
def semantic_search(
    query: str = typer.Argument(..., help="Text query for semantic matching."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return."),
    filters: str | None = typer.Option(
        None,
        "--filters",
        "-f",
        help='JSON string for metadata filtering, e.g., \'{"space_key": "DEV"}\'.',
    ),
) -> None:
    search_service: SearchService = _get_search_service()
    print_status("Performing Semantic Search...", "info")

    parsed_filters: dict[str, Any] | None = None
    if filters:
        try:
            parsed_filters = json.loads(filters)
            if not isinstance(parsed_filters, dict):
                raise SearchParameterError(
                    "Filters must be a valid JSON object string."
                )
            print(f"Applying filters: {parsed_filters}")
        except json.JSONDecodeError as e:
            raise SearchParameterError(f"Invalid JSON in filters string: {e}")

    results, took_ms = search_service.search_semantic(
        query=query, top_k=top_k, filters=parsed_filters
    )

    print_semantic_search_results(results=results, query=query, took_ms=took_ms)
