import logging
from functools import wraps

from fastapi import APIRouter, Depends, HTTPException, Request

from confluence_gateway.api.dependencies import get_generation_service
from confluence_gateway.api.schemas.requests import GenerateAnswerRequest
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    GenerateAnswerResponse,
    SourceDocument,
)
from confluence_gateway.core.exceptions import (
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    GenerationError,
    SearchParameterError,
    SemanticSearchError,
)
from confluence_gateway.services.generation import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter()


def handle_generation_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except GenerationError as e:
            error = ErrorResponse(
                code=500,
                message=f"Generation failed: {str(e)}",
                details={"type": "generation_error"},
            )
            raise HTTPException(status_code=500, detail=error.model_dump())
        except SemanticSearchError as e:
            error = ErrorResponse(
                code=500,
                message=f"Context retrieval failed: {str(e)}",
                details={"type": "semantic_search_error"},
            )
            raise HTTPException(status_code=500, detail=error.model_dump())
        except SearchParameterError as e:
            error = ErrorResponse(
                code=400, message=str(e), details={"type": "search_parameter_error"}
            )
            raise HTTPException(status_code=400, detail=error.model_dump())
        except ConfluenceAuthenticationError as e:
            error = ErrorResponse(
                code=401, message=str(e), details={"type": "authentication_error"}
            )
            raise HTTPException(status_code=401, detail=error.model_dump())
        except ConfluenceConnectionError as e:
            error = ErrorResponse(
                code=503,
                message=f"Confluence connection error during context retrieval: {str(e)}",
                details={
                    "type": "connection_error",
                    "cause": str(getattr(e, "cause", "")),
                },
            )
            raise HTTPException(status_code=503, detail=error.model_dump())
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True)
            error = ErrorResponse(
                code=500,
                message=f"Unexpected server error during generation: {str(e)}",
                details={"type": "server_error"},
            )
            raise HTTPException(status_code=500, detail=error.model_dump())

    return wrapper


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
@handle_generation_exceptions
async def generate_answer(
    request: Request,
    gen_request: GenerateAnswerRequest,
    generation_service: GenerationService = Depends(get_generation_service),
):
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
