import logging

from fastapi import APIRouter, BackgroundTasks, Depends, status

from confluence_gateway.api.dependencies import get_indexing_service
from confluence_gateway.api.schemas.requests import IndexingTriggerRequest
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    IndexingStatusResponse,
)
from confluence_gateway.core.exception_mapping import APIExceptionHandler
from confluence_gateway.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/trigger",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Background Indexing",
    description="Starts the Confluence content indexing process in the background. Accepts an optional list of space keys.",
    responses={
        409: {"model": ErrorResponse, "description": "Indexing is already running"},
        503: {
            "model": ErrorResponse,
            "description": "Indexing service not available (config error)",
        },
    },
)
@APIExceptionHandler.handle_exceptions
async def trigger_indexing(
    request: IndexingTriggerRequest,
    background_tasks: BackgroundTasks,
    indexing_service: IndexingService | None = Depends(get_indexing_service),
) -> IndexingStatusResponse:
    if indexing_service is None:
        logger.error("Indexing trigger failed: IndexingService is not available.")
        from confluence_gateway.core.exceptions import ConfluenceGatewayError

        raise ConfluenceGatewayError(
            "Indexing service is not configured or failed to initialize."
        )

    current_status = indexing_service.status["status"]
    if current_status == "running":
        logger.warning("Indexing trigger failed: Process is already running.")
        from confluence_gateway.core.exceptions import ConfluenceGatewayError

        raise ConfluenceGatewayError("Indexing process is already running.")

    if request.index_all:
        target_description = "All accessible spaces"
    elif request.space_keys:
        target_description = f"{request.space_keys}"
    else:
        target_description = "Configured spaces"

    logger.info(f"Adding indexing task to background. Target: {target_description}")
    background_tasks.add_task(
        indexing_service.run_indexing, request.space_keys, request.index_all
    )

    # Return current status after triggering
    status_data = indexing_service.status
    return IndexingStatusResponse(**status_data)


@router.get(
    "/status",
    response_model=IndexingStatusResponse,
    summary="Get Indexing Status",
    description="Retrieves the current status and details of the last indexing run.",
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Indexing service not available (config error)",
        },
    },
)
@APIExceptionHandler.handle_exceptions
async def get_indexing_status(
    indexing_service: IndexingService | None = Depends(get_indexing_service),
) -> IndexingStatusResponse:
    if indexing_service is None:
        return IndexingStatusResponse(
            status="failure",
            last_run_start_time=None,
            last_run_end_time=None,
            last_error_message="Indexing service is not configured or failed to initialize.",
        )

    status_data = indexing_service.status
    return IndexingStatusResponse(**status_data)
