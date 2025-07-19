from fastapi import APIRouter, Depends, Path

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import ConfluenceSpace
from confluence_gateway.api.dependencies import get_confluence_client
from confluence_gateway.api.schemas.responses import SpaceInfo
from confluence_gateway.core.exception_mapping import APIExceptionHandler
from confluence_gateway.core.transformers import SpaceTransformer

router = APIRouter()


def _convert_to_space_info(space: ConfluenceSpace) -> SpaceInfo:
    """Convert ConfluenceSpace model to SpaceInfo response model."""
    # Use shared transformer to extract space data
    space_data = SpaceTransformer.extract_space_data(space)

    return SpaceInfo(
        id=space_data["id"],
        key=space_data["key"],
        name=space_data["name"],
        type=space_data["type"],
        description=space_data["description"],
        created_at=space_data["created_at"],
        updated_at=space_data["updated_at"],
    )


@router.get("/", response_model=list[SpaceInfo])
@APIExceptionHandler.handle_exceptions
async def list_spaces(
    client: ConfluenceClient | None = Depends(get_confluence_client),
) -> list[SpaceInfo]:
    """
    List all Confluence spaces.

    Returns a list of all accessible Confluence spaces with their details.
    """
    if client is None:
        # Return mock spaces for development mode
        from datetime import datetime, timezone

        return [
            SpaceInfo(
                id="mock-123",
                key="MOCK",
                name="Mock Space",
                type="global",
                description="This is a mock space for development mode",
                homepage_url="/spaces/MOCK",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]

    spaces = client.list_all_spaces()
    return [_convert_to_space_info(space) for space in spaces]


@router.get("/{space_key}", response_model=SpaceInfo)
@APIExceptionHandler.handle_exceptions
async def get_space(
    space_key: str = Path(..., description="The space key to retrieve"),
    client: ConfluenceClient | None = Depends(get_confluence_client),
) -> SpaceInfo:
    """
    Get detailed information about a specific Confluence space.

    Args:
        space_key: The unique key identifying the space

    Returns:
        Detailed information about the requested space
    """
    if client is None:
        # Return mock space for development mode
        from datetime import datetime, timezone

        return SpaceInfo(
            id=f"mock-{space_key}-123",
            key=space_key,
            name=f"Mock Space ({space_key})",
            type="global",
            description=f"This is a mock space for development mode with key: {space_key}",
            homepage_url=f"/spaces/{space_key}",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    space = client.get_space(space_key)
    return _convert_to_space_info(space)
