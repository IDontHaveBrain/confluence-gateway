import logging

from fastapi import APIRouter, Depends, Request

from confluence_gateway.api.dependencies import get_generation_service
from confluence_gateway.api.schemas.requests import GenerateAnswerRequest
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    GenerateAnswerResponse,
    SourceDocument,
)
from confluence_gateway.core.exception_mapping import APIExceptionHandler
from confluence_gateway.services.generation import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/answer",
    response_model=GenerateAnswerResponse,
    summary="Generate Answer using RAG",
    description="Generates an answer to a query by retrieving relevant Confluence content and using an LLM.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Generation or retrieval error"},
        501: {
            "model": ErrorResponse,
            "description": "Generation feature is disabled",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service unavailable (e.g., LLM connection issue, missing dependency)",
        },
    },
)
@APIExceptionHandler.handle_exceptions
async def generate_answer(
    request: Request,
    gen_request: GenerateAnswerRequest,
    generation_service: GenerationService = Depends(get_generation_service),
) -> GenerateAnswerResponse:
    logger.info(
        f"Received generation request: query='{gen_request.query[:50]}...', top_k={gen_request.top_k_retrieval or '(default)'}, filters={gen_request.filters}"
    )

    top_k = (
        gen_request.top_k_retrieval
        if gen_request.top_k_retrieval is not None
        else 5  # Default from GenerationService.generate_answer signature
    )

    answer, retrieved_results = await generation_service.generate_answer(
        query=gen_request.query,
        top_k_retrieval=top_k,
        filters=gen_request.filters,
    )

    source_docs = [
        SourceDocument.from_vector_search_item(item) for item in retrieved_results
    ]

    return GenerateAnswerResponse(answer=answer, sources=source_docs)
