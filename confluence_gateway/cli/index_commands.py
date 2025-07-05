import logging

import typer

from confluence_gateway.api.schemas.responses import IndexingStatusResponse
from confluence_gateway.cli.common import (
    handle_cli_errors,
    print_indexing_status,
    print_status,
)
from confluence_gateway.cli.dependencies import _get_indexing_service
from confluence_gateway.services.indexing import IndexingService

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Manage the content indexing process for semantic search.",
)


@app.command("trigger", help="Trigger indexing process (runs synchronously).")
@handle_cli_errors
def trigger_indexing(
    space_keys: list[str] | None = typer.Option(
        None,
        "--space",
        "-s",
        help="Specific space keys to index (optional, defaults to configured spaces).",
        show_default=False,
    ),
):
    indexing_service: IndexingService | None = _get_indexing_service()

    if indexing_service is None:
        raise typer.Exit(code=1)

    current_status = indexing_service.status["status"]
    if current_status == "running":
        print_status("Indexing process is already running.", "warning")
        raise typer.Exit(code=0)

    print_status(
        f"Starting synchronous indexing... Target spaces: {space_keys or 'Configured'}",
        "info",
    )
    print_status("(This may take a while depending on the content size)", "dim")

    try:
        indexing_service._run_indexing_sync(space_keys=space_keys)
        print_status("Synchronous indexing completed successfully.", "success")
    except Exception as e:
        logger.error(f"Synchronous indexing failed: {e}", exc_info=True)
        raise


@app.command("status", help="Get the current status of the indexing service.")
@handle_cli_errors
def get_status():
    indexing_service: IndexingService | None = _get_indexing_service()

    if indexing_service is None:
        status_response = IndexingStatusResponse(
            status="failure",
            last_run_start_time=None,
            last_run_end_time=None,
            last_error_message="Indexing service unavailable (Vector DB or Embedding likely disabled/misconfigured).",
        )
        print_indexing_status(status_response)
        return

    status_dict = indexing_service.status
    status_response = IndexingStatusResponse(**status_dict)
    print_indexing_status(status_response)
