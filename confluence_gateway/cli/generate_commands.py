import asyncio
import logging
from typing import Any, cast

import typer

from confluence_gateway.cli.common import (
    handle_cli_errors,
    print_generated_answer,
    print_status,
)
from confluence_gateway.core.exceptions import (
    GenerationError,
    SearchParameterError,
    SemanticSearchError,
)
from confluence_gateway.core.validation import ValidationUtils

logger = logging.getLogger(__name__)

app = typer.Typer(
    no_args_is_help=True,
    help="Generate answers using Retrieval-Augmented Generation (RAG).",
)


@app.command("answer", help="Generate an answer using RAG based on Confluence content.")
@handle_cli_errors
def generate_answer_command(
    query: str = typer.Argument(..., help="The question to ask."),
    top_k_retrieval: int | None = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of context documents to retrieve.",
        show_default=True,
    ),
    filters: str | None = typer.Option(
        None,
        "--filters",
        "-f",
        help='JSON string for metadata filtering during retrieval, e.g., \'{"space_key": "DEV"}\'.',
        show_default=False,
    ),
) -> None:
    from confluence_gateway.api.schemas.responses import SourceDocument
    from confluence_gateway.cli.dependencies import (
        StubGenerationService,
        _get_generation_service,
    )
    from confluence_gateway.services.generation import GenerationService

    generation_service: GenerationService | StubGenerationService = (
        _get_generation_service()
    )

    print_status("Generating answer using RAG...", "info")

    parsed_filters: dict[str, Any] | None = None
    if filters:
        parsed_filters = ValidationUtils.validate_json_string(filters, "filters")
        print(f"Applying retrieval filters: {parsed_filters}")

    try:
        logger.info(
            f"Calling generation service for query: '{query[:50]}...', top_k={top_k_retrieval}, filters={parsed_filters}"
        )
        result = asyncio.run(
            generation_service.generate_answer(
                query=query,
                top_k_retrieval=top_k_retrieval or 5,
                filters=parsed_filters,
            )
        )
        answer = result[0]
        retrieved_results = cast(list[Any], result[1])
        logger.info("Generation service call completed.")

        source_docs = [
            SourceDocument.from_vector_search_item(item) for item in retrieved_results
        ]

        print_generated_answer(answer=answer, sources=source_docs)

    except (GenerationError, SemanticSearchError, SearchParameterError) as e:
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error during generate_answer call: {e}", exc_info=True
        )
        raise RuntimeError(f"An unexpected error occurred during generation: {e}")
