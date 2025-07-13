import json
import logging
from functools import wraps
from typing import Any

import typer

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.api.schemas.responses import (
    IndexingStatusResponse,
    SearchResultItem,
    SourceDocument,
)
from confluence_gateway.core.exceptions import ConfluenceGatewayError

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
    results: list[SearchResultItem], total: int, start: int, limit: int, took_ms: float
) -> None:
    # Convert results to JSON-serializable format
    results_data = []
    for item in results:
        result_dict = {
            "id": item.id,
            "title": item.title,
            "type": item.type,
            "space_key": item.space_key,
            "space_name": item.space_name,
            "last_modified": item.last_modified.isoformat()
            if item.last_modified
            else None,
            "url": item.url,
        }
        if hasattr(item, "content") and item.content:
            result_dict["content"] = item.content
        if hasattr(item, "excerpt") and item.excerpt:
            result_dict["excerpt"] = item.excerpt
        results_data.append(result_dict)

    result = {
        "results": results_data,
        "total": total,
        "start": start,
        "limit": limit,
        "count": len(results),
        "took_ms": took_ms,
    }
    print(json.dumps(result, indent=2))


def print_semantic_search_results(
    results: list[VectorSearchResultItem], query: str, took_ms: float
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


def print_indexing_status(status: IndexingStatusResponse) -> None:
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


def print_generated_answer(answer: str, sources: list[SourceDocument]) -> None:
    # Convert sources to JSON-serializable format
    sources_data = []
    for source in sources:
        source_dict = {
            "id": source.id,
            "score": source.score,
            "title": source.title,
            "space_key": source.space_key,
            "url": source.url,
        }
        if hasattr(source, "content") and source.content:
            source_dict["content"] = source.content
        sources_data.append(source_dict)

    result = {
        "answer": answer,
        "sources": sources_data,
        "source_count": len(sources),
    }
    print(json.dumps(result, indent=2))


def handle_cli_errors(func: Any) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ConfluenceGatewayError as e:
            error_type = type(e).__name__
            logger.error(f"CLI Error ({error_type}): {e}", exc_info=True)
            print_status(f"Error ({error_type}): {str(e)}", "error")
            raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except Exception as e:
            logger.error(f"Unexpected CLI Error: {e}", exc_info=True)
            print_status(f"Unexpected Error: {str(e)}", "error")
            print_status("Check logs for more details.", "dim")
            raise typer.Exit(code=1)

    return wrapper
