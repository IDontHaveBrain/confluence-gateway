"""Unified data transformation utilities for API and CLI interfaces."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SearchResultTransformer:
    """Unified search result data transformation logic."""

    @staticmethod
    def extract_and_build_search_result_data(item: Any, client: Any) -> dict[str, Any]:
        """Extract content fields and build normalized search result data."""
        # Extract content fields once to avoid multiple calls
        if client is None:
            # Handle dev mode where client is None - extract fields directly from item
            extracted = {
                "id": getattr(item, "id", ""),
                "title": getattr(item, "title", ""),
                "type": getattr(item, "type", ""),
                "space": getattr(item, "space", {}),
                "content": getattr(item, "content", {}),
                "excerpt": "",
                "url": getattr(item, "url", ""),
                "_links": getattr(item, "_links", {}),
                "lastModified": None,
                "created": None,
            }
        else:
            extracted = client.extract_content_fields(item)

        # Construct URL with fallback logic
        url = SearchResultTransformer._build_url(extracted, client, item)

        # Handle datetime fields with fallbacks
        last_modified = SearchResultTransformer._extract_datetime(extracted, item)

        return {
            "id": extracted.get("id", str(getattr(item, "id", ""))),
            "title": extracted.get("title", "Title not available"),
            "type": extracted.get("type", str(getattr(item, "content_type", "page"))),
            "space_key": extracted.get("space_key", ""),
            "space_name": extracted.get("space_name", "Space name not available"),
            "url": url,
            "excerpt": getattr(item, "excerpt", None),
            "last_modified": last_modified,
        }

    @staticmethod
    def build_search_result_items_from_pages(
        pages: list[Any], client: Any
    ) -> list[dict[str, Any]]:
        """Transform a list of pages into normalized search result data."""
        return [
            SearchResultTransformer.extract_and_build_search_result_data(page, client)
            for page in pages
        ]

    @staticmethod
    def build_search_result_items_from_search_result(
        search_result: Any, search_service: Any
    ) -> list[dict[str, Any]]:
        """Transform search result object into normalized search result data."""
        return [
            SearchResultTransformer.extract_and_build_search_result_data(
                item, search_service.client
            )
            for item in search_result.results.results
        ]

    @staticmethod
    def _build_url(extracted: dict[str, Any], client: Any, item: Any) -> str:
        """Build URL with consistent fallback logic."""
        # Try to get URL from extracted fields first
        url = extracted.get("url")
        if url:
            return str(url)

        # Fallback: construct URL from base URL and identifiers
        space_key = extracted.get("space_key")
        page_id = extracted.get("id", str(getattr(item, "id", "")))
        base_url = getattr(client, "base_url", None) if client else None

        if base_url and space_key and page_id:
            return f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"

        return "URL not available"

    @staticmethod
    def _extract_datetime(extracted: dict[str, Any], item: Any) -> datetime:
        """Extract datetime with consistent fallback logic."""
        # Try multiple datetime fields in order of preference
        datetime_candidates = [
            extracted.get("updated_at"),
            extracted.get("created_at"),
            getattr(item, "updated_at", None),
            getattr(item, "created_at", None),
        ]

        for dt in datetime_candidates:
            if dt:
                if isinstance(dt, datetime):
                    return dt
                # If dt is not datetime, try to parse it or skip
                continue

        # Final fallback to current time
        return datetime.now()


class SpaceTransformer:
    """Unified space information transformation logic."""

    @staticmethod
    def extract_space_data(space: Any) -> dict[str, Any]:
        """Extract and normalize space information."""
        # Handle description extraction with multiple fallback strategies
        description = SpaceTransformer._extract_description(space)

        # Handle type extraction
        space_type = SpaceTransformer._extract_type(space)

        # Handle name/title with fallback
        space_name = SpaceTransformer._extract_name(space)

        return {
            "id": getattr(space, "id", ""),
            "key": getattr(space, "key", ""),
            "name": space_name,
            "title": getattr(
                space, "title", space_name
            ),  # Include both for CLI compatibility
            "type": space_type,
            "description": description,
            "created_at": getattr(space, "created_at", None),
            "updated_at": getattr(space, "updated_at", None),
        }

    @staticmethod
    def extract_spaces_data(spaces: list[Any]) -> list[dict[str, Any]]:
        """Transform a list of spaces into normalized data."""
        return [SpaceTransformer.extract_space_data(space) for space in spaces]

    @staticmethod
    def _extract_description(space: Any) -> str | None:
        """Extract description with multiple fallback strategies."""
        # Try description_text attribute first (most common)
        if hasattr(space, "description_text") and space.description_text:
            return str(space.description_text)

        # Try complex description dict structure
        if (
            hasattr(space, "description")
            and space.description
            and isinstance(space.description, dict)
        ):
            if "plain" in space.description and isinstance(
                space.description["plain"], dict
            ):
                return space.description["plain"].get("value")

        return None

    @staticmethod
    def _extract_type(space: Any) -> str:
        """Extract space type with fallback."""
        if hasattr(space, "type") and space.type:
            # Handle enum-like type objects
            if hasattr(space.type, "value"):
                return str(space.type.value)
            return str(space.type)
        return "unknown"

    @staticmethod
    def _extract_name(space: Any) -> str:
        """Extract space name with title fallback."""
        name = getattr(space, "name", None)
        if name:
            return str(name)

        title = getattr(space, "title", None)
        if title:
            return str(title)

        return "Unknown Space"


class PaginationTransformer:
    """Unified pagination response building logic."""

    @staticmethod
    def build_pagination_data(
        total: int,
        start: int,
        limit: int,
        current_count: int | None = None,
        base_url: str | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build consistent pagination metadata."""
        if current_count is None:
            current_count = min(limit, total - start)

        # Calculate pagination values
        page_count = (total + limit - 1) // limit if limit > 0 else 1
        current_page = (start // limit) + 1 if limit > 0 else 1
        has_more = start + current_count < total

        pagination_data: dict[str, Any] = {
            "total": total,
            "start": start,
            "limit": limit,
            "count": current_count,
            "page_count": page_count,
            "current_page": current_page,
            "has_more": has_more,
        }

        # Add navigation links if base URL provided
        if base_url and query_params:
            pagination_data["links"] = PaginationTransformer._build_navigation_links(
                base_url, query_params, start, limit, total
            )

        return pagination_data

    @staticmethod
    def _build_navigation_links(
        base_url: str, query_params: dict[str, Any], start: int, limit: int, total: int
    ) -> dict[str, str | None]:
        """Build navigation links for pagination."""
        from urllib.parse import urlencode

        def build_link(new_start: int) -> str:
            params = query_params.copy()
            params["start"] = new_start
            params["limit"] = limit
            return f"{base_url}?{urlencode(params)}"

        links = {
            "self": build_link(start),
            "first": build_link(0) if start > 0 else None,
            "previous": build_link(max(0, start - limit)) if start > 0 else None,
            "next": build_link(start + limit) if start + limit < total else None,
            "last": build_link(((total - 1) // limit) * limit)
            if start + limit < total
            else None,
        }

        return links


class ResponseTransformer:
    """Unified response building utilities."""

    @staticmethod
    def build_search_response_data(
        search_result: Any,
        search_service: Any,
        took_ms: float,
        start: int = 0,
        limit: int = 25,
        base_url: str | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build complete search response data structure."""
        # Transform search result items
        search_items_data = (
            SearchResultTransformer.build_search_result_items_from_search_result(
                search_result, search_service
            )
        )

        # Extract total from search result
        total = getattr(search_result.results, "total", len(search_items_data))

        # Build pagination data
        pagination_data = PaginationTransformer.build_pagination_data(
            total=total,
            start=start,
            limit=limit,
            current_count=len(search_items_data),
            base_url=base_url,
            query_params=query_params,
        )

        # Combine all data
        response_data = {
            "results": search_items_data,
            "took_ms": took_ms,
            **pagination_data,
        }

        return response_data

    @staticmethod
    def build_generation_response_data(
        answer: str,
        source_documents: list[Any],
        took_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build generation response data structure."""
        # Transform source documents to consistent format
        sources_data = [
            {
                "id": getattr(doc, "id", ""),
                "title": getattr(doc, "title", ""),
                "url": getattr(doc, "url", ""),
                "space_key": getattr(doc, "space_key", ""),
                "excerpt": getattr(doc, "content", "")[:200] + "..."
                if getattr(doc, "content", "")
                else None,
            }
            for doc in source_documents
        ]

        response_data = {
            "answer": answer,
            "sources": sources_data,
            "took_ms": took_ms,
            "metadata": metadata or {},
        }

        return response_data

    @staticmethod
    def build_indexing_response_data(
        status: str, message: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build indexing operation response data."""
        return {
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }


class DataFormatConverter:
    """Utilities for converting between different data formats."""

    @staticmethod
    def to_json_serializable(data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {
                key: DataFormatConverter.to_json_serializable(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [DataFormatConverter.to_json_serializable(item) for item in data]
        elif hasattr(data, "model_dump"):  # Pydantic model
            # Use model_dump with mode='json' to properly serialize datetime objects
            dumped_data = data.model_dump(mode="json")
            # Recursively process in case there are nested structures that need conversion
            return DataFormatConverter.to_json_serializable(dumped_data)
        elif hasattr(data, "__dict__"):  # Object with attributes
            return DataFormatConverter.to_json_serializable(data.__dict__)
        else:
            return data

    @staticmethod
    def extract_field_with_fallbacks(
        obj: Any, field_names: list[str], default: Any = None
    ) -> Any:
        """Extract field from object with multiple fallback field names."""
        for field_name in field_names:
            if hasattr(obj, field_name):
                value = getattr(obj, field_name)
                if value is not None:
                    return value
        return default
