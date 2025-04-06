import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from confluence_gateway.api.dependencies import get_indexing_service
from confluence_gateway.api.schemas.requests import IndexingTriggerRequest
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    IndexingStatusResponse,
)
from confluence_gateway.services.indexing import IndexingService

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
async def trigger_indexing(
    request: IndexingTriggerRequest,
    background_tasks: BackgroundTasks,
    indexing_service: Optional[IndexingService] = Depends(get_indexing_service),
):
    if indexing_service is None:
        logger.error("Indexing trigger failed: IndexingService is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                status="error",
                code=503,
                message="Indexing service is not configured or failed to initialize.",
                details={"type": "service_unavailable"},
            ).model_dump(),
        )

    current_status = indexing_service.status["status"]
    if current_status == "running":
        logger.warning("Indexing trigger failed: Process is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                status="error",
                code=409,
                message="Indexing process is already running.",
                details={"type": "indexing_in_progress"},
            ).model_dump(),
        )

    logger.info(
        f"Adding indexing task to background. Target spaces: {request.space_keys or 'Configured'}"
    )
    background_tasks.add_task(indexing_service.run_indexing, request.space_keys)

    return {"message": "Indexing task accepted and started in background."}


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
async def get_indexing_status(
    indexing_service: Optional[IndexingService] = Depends(get_indexing_service),
):
    if indexing_service is None:
        return IndexingStatusResponse(
            status="failure",
            last_run_start_time=None,
            last_run_end_time=None,
            last_error_message="Indexing service is not configured or failed to initialize.",
        )

    status_data = indexing_service.status
    return IndexingStatusResponse(**status_data)
