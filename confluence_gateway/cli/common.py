import json
import logging
from typing import Any

from confluence_gateway.core.exception_mapping import CLIExceptionHandler
from confluence_gateway.core.transformers import (
    DataFormatConverter,
    PaginationTransformer,
)

logger = logging.getLogger(__name__)


def print_status(message: str, status_type: str = "info") -> None:
    """Print a status message with optional prefix."""
    prefix_map = {
        "info": "[INFO]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "success": "[OK]",
        "dim": "",
    }
    prefix = prefix_map.get(status_type, "")
    if prefix:
        print(f"{prefix} {message}")
    else:
        print(message)


def print_search_results(
    results: list[Any], total: int, start: int, limit: int, took_ms: float
) -> None:
    # Use shared transformer to build consistent response format
    pagination_data = PaginationTransformer.build_pagination_data(
        total=total, start=start, limit=limit, current_count=len(results)
    )

    # Convert results to JSON-serializable format using shared utility
    results_data = []
    for item in results:
        item_dict = DataFormatConverter.to_json_serializable(item)
        results_data.append(item_dict)

    result = {"results": results_data, "took_ms": took_ms, **pagination_data}
    print(json.dumps(result, indent=2))


def print_semantic_search_results(
    results: list[Any], query: str, took_ms: float
) -> None:
    # Convert results to JSON-serializable format
    results_data = []
    for item in results:
        metadata = item.metadata or {}
        result_dict = {
            "id": item.id,
            "score": item.score,
            "text": item.text,
            "metadata": metadata,
        }
        results_data.append(result_dict)

    result = {
        "query": query,
        "results": results_data,
        "count": len(results),
        "took_ms": took_ms,
    }
    print(json.dumps(result, indent=2))


def print_indexing_status(status: Any) -> None:
    # Convert status to JSON-serializable format
    status_dict = {
        "status": status.status,
        "last_run_start_time": status.last_run_start_time.isoformat()
        if status.last_run_start_time
        else None,
        "last_run_end_time": status.last_run_end_time.isoformat()
        if status.last_run_end_time
        else None,
        "last_error_message": status.last_error_message,
    }
    print(json.dumps(status_dict, indent=2))


def print_generated_answer(answer: str, sources: list[Any]) -> None:
    # Convert sources to JSON-serializable format using shared utility
    sources_data = [
        DataFormatConverter.to_json_serializable(source) for source in sources
    ]

    result = {
        "answer": answer,
        "sources": sources_data,
        "source_count": len(sources),
    }
    print(json.dumps(result, indent=2))


def handle_cli_errors(func: Any) -> Any:
    """Legacy wrapper - redirects to shared exception handler."""
    return CLIExceptionHandler.handle_exceptions(func)


def handle_cli_errors_verbose(func: Any) -> Any:
    """Legacy wrapper with verbose output - redirects to shared exception handler."""
    return CLIExceptionHandler.handle_exceptions_verbose(func)
