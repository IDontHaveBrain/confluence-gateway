from typing import Any, cast

import typer

from confluence_gateway.cli.common import (
    handle_cli_errors,
    print_search_results,
    print_semantic_search_results,
    print_status,
)
from confluence_gateway.core.transformers import SearchResultTransformer
from confluence_gateway.core.validation import ValidationUtils

app = typer.Typer(
    no_args_is_help=True,
    help="Search Confluence content using various methods.",
)


def _convert_to_search_result_items(pages: list[Any], client: Any) -> list[Any]:
    from confluence_gateway.api.schemas.responses import SearchResultItem

    # Use shared transformer to get normalized data
    items_data = SearchResultTransformer.build_search_result_items_from_pages(
        pages, client
    )

    # Convert to SearchResultItem objects
    items = [SearchResultItem(**item_data) for item_data in items_data]

    return items


@app.command("text", help="Search using text query.")
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
    from confluence_gateway.cli.dependencies import (
        StubSearchService,
        _get_search_service,
    )
    from confluence_gateway.services.search import EnhancedSearchResult, SearchService

    search_service: SearchService | StubSearchService = _get_search_service()

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
            min_relevance=min_relevance or 0.0,
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


@app.command("hybrid", help="Search using hybrid (keyword + semantic + RRF).")
@handle_cli_errors
def hybrid_search(
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
) -> None:
    from confluence_gateway.cli.dependencies import (
        StubSearchService,
        _get_search_service,
    )
    from confluence_gateway.services.search import EnhancedSearchResult, SearchService

    search_service: SearchService | StubSearchService = _get_search_service()

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
    from confluence_gateway.cli.dependencies import (
        StubSearchService,
        _get_search_service,
    )
    from confluence_gateway.services.search import EnhancedSearchResult, SearchService

    search_service: SearchService | StubSearchService = _get_search_service()
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
    from confluence_gateway.cli.dependencies import (
        StubSearchService,
        _get_search_service,
    )
    from confluence_gateway.services.search import SearchService

    search_service: SearchService | StubSearchService = _get_search_service()
    print_status("Performing Semantic Search...", "info")

    parsed_filters: dict[str, Any] | None = None
    if filters:
        parsed_filters = ValidationUtils.validate_json_string(filters, "filters")
        print(f"Applying filters: {parsed_filters}")

    results, took_ms = search_service.search_semantic(
        query=query, top_k=top_k, filters=parsed_filters
    )

    print_semantic_search_results(results=results, query=query, took_ms=took_ms)
