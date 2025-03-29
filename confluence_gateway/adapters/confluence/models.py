from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Enum representing the different types of content in Confluence."""

    PAGE = "page"
    BLOGPOST = "blogpost"
    ATTACHMENT = "attachment"
    COMMENT = "comment"


class SpaceType(str, Enum):
    """Enum representing the different types of spaces in Confluence."""

    GLOBAL = "global"
    PERSONAL = "personal"


class ConfluenceObject(BaseModel):
    """Base class for all Confluence objects."""

    id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def __init__(self, **data):
        """Initialize object, handling Confluence API date fields."""
        # Map Confluence API field names to our model field names
        if "created" in data and data["created"] and "created_at" not in data:
            data["created_at"] = data["created"]

        if "updated" in data and data["updated"] and "updated_at" not in data:
            data["updated_at"] = data["updated"]

        super().__init__(**data)


class ConfluenceSpace(ConfluenceObject):
    """Representation of a Confluence space."""

    key: str
    name: Optional[str] = None
    description: Optional[dict[str, Any]] = None
    type: Optional[SpaceType] = None

    def __init__(self, **data):
        """Initialize space object, handling Confluence API field structure."""
        # Handle name/title ambiguity in Confluence API
        if "name" in data and "title" not in data:
            data["title"] = data["name"]

        # Handle description structure which can be complex in Confluence API
        if "description" in data and isinstance(data["description"], dict):
            if "plain" in data["description"]:
                plain_desc = data["description"]["plain"]
                if (
                    plain_desc
                    and isinstance(plain_desc, dict)
                    and "value" in plain_desc
                ):
                    data["description"] = plain_desc

        super().__init__(**data)


class BodyContent(BaseModel):
    """Content of a Confluence page/blog post."""

    view: Optional[dict[str, Any]] = None
    storage: Optional[dict[str, Any]] = None
    plain: Optional[dict[str, Any]] = None

    model_config = {
        "populate_by_name": True,
    }


class Version(BaseModel):
    """Version information for Confluence content."""

    number: int
    when: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
    }


class ConfluencePage(ConfluenceObject):
    """Representation of a Confluence page, blog post, or other content."""

    space: Optional[Union[ConfluenceSpace, dict[str, Any]]] = None
    content_type: ContentType = ContentType.PAGE
    body: Optional[BodyContent] = None
    version: Optional[Version] = None
    status: Optional[str] = None

    def __init__(self, **data):
        """Initialize page object, handling Confluence API field structure."""
        # Extract the content type if provided
        if "type" in data and "content_type" not in data:
            data["content_type"] = data["type"]

        # Handle body structure
        if "body" in data and isinstance(data["body"], dict):
            data["body"] = BodyContent(**data["body"])

        # Handle version structure
        if "version" in data and isinstance(data["version"], dict):
            data["version"] = Version(**data["version"])

        # Handle space reference
        if "space" in data and isinstance(data["space"], dict):
            # Don't convert to ConfluenceSpace yet to avoid circular import issues
            # We'll convert later when needed
            pass

        super().__init__(**data)

    @property
    def html_content(self) -> Optional[str]:
        """Get HTML content from the page body if available."""
        if self.body and self.body.view and "value" in self.body.view:
            return self.body.view["value"]
        return None

    @property
    def storage_content(self) -> Optional[str]:
        """Get storage content from the page body if available."""
        if self.body and self.body.storage and "value" in self.body.storage:
            return self.body.storage["value"]
        return None

    @property
    def plain_content(self) -> Optional[str]:
        """Get plain text content from the page body if available."""
        if self.body and self.body.plain and "value" in self.body.plain:
            return self.body.plain["value"]
        return None


class SearchResult(BaseModel):
    """Container for Confluence search results."""

    total_size: int = Field(0, description="Total number of results available")
    start: int = Field(0, description="Starting index of results")
    limit: int = Field(0, description="Maximum number of results returned")
    results: list[ConfluencePage] = Field(
        default_factory=list, description="Search result items"
    )

    model_config = {
        "populate_by_name": True,
    }

    def __init__(self, **data):
        """Initialize search result, handling Confluence API response structure."""
        # Map API field names to our model field names
        if "size" in data and "total_size" not in data:
            data["total_size"] = data["size"]

        # Transform result items
        if "results" in data and isinstance(data["results"], list):
            transformed_results = []
            for item in data["results"]:
                if isinstance(item, dict):
                    transformed_results.append(ConfluencePage(**item))
                elif isinstance(item, ConfluencePage):
                    transformed_results.append(item)
            data["results"] = transformed_results

        super().__init__(**data)
