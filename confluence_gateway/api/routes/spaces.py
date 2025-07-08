from functools import wraps
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import ConfluenceSpace
from confluence_gateway.api.dependencies import get_confluence_client
from confluence_gateway.api.schemas.responses import ErrorResponse, SpaceInfo
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
)

router = APIRouter()


def handle_confluence_exceptions(func: Any) -> Any:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ConfluenceAuthenticationError as e:
            error = ErrorResponse(
                code=401, message=str(e), details={"type": "authentication_error"}
            )
            raise HTTPException(status_code=401, detail=error.model_dump())
        except ConfluenceConnectionError as e:
            error = ErrorResponse(
                code=503, message=str(e), details={"type": "connection_error"}
            )
            raise HTTPException(status_code=503, detail=error.model_dump())
        except ConfluenceAPIError as e:
            error = ErrorResponse(
                code=e.status_code or 500,
                message=e.error_message,
                details={"type": "api_error"},
            )
            raise HTTPException(
                status_code=e.status_code or 500, detail=error.model_dump()
            )
        except Exception as e:
            error = ErrorResponse(
                code=500,
                message=f"Unexpected error: {str(e)}",
                details={"type": "unexpected_error"},
            )
            raise HTTPException(status_code=500, detail=error.model_dump())

    return wrapper


def _convert_to_space_info(space: ConfluenceSpace) -> SpaceInfo:
    """Convert ConfluenceSpace model to SpaceInfo response model."""
    description = None
    if hasattr(space, "description_text"):
        description = space.description_text
    elif space.description and isinstance(space.description, dict):
        if "plain" in space.description and isinstance(
            space.description["plain"], dict
        ):
            description = space.description["plain"].get("value")

    space_type = space.type.value if space.type else "unknown"
    space_name = space.name or space.title

    return SpaceInfo(
        id=space.id,
        key=space.key,
        name=space_name,
        type=space_type,
        description=description,
        created_at=space.created_at,
        updated_at=space.updated_at,
    )


@router.get("/", response_model=list[SpaceInfo])
@handle_confluence_exceptions
async def list_spaces(
    client: ConfluenceClient = Depends(get_confluence_client),
) -> list[SpaceInfo]:
    """
    List all Confluence spaces.

    Returns a list of all accessible Confluence spaces with their details.
    """
    spaces = client.list_all_spaces()
    return [_convert_to_space_info(space) for space in spaces]


@router.get("/{space_key}", response_model=SpaceInfo)
@handle_confluence_exceptions
async def get_space(
    space_key: str = Path(..., description="The space key to retrieve"),
    client: ConfluenceClient = Depends(get_confluence_client),
) -> SpaceInfo:
    """
    Get detailed information about a specific Confluence space.

    Args:
        space_key: The unique key identifying the space

    Returns:
        Detailed information about the requested space
    """
    space = client.get_space(space_key)
    return _convert_to_space_info(space)
