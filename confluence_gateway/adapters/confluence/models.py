from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    PAGE = "page"
    BLOGPOST = "blogpost"
    ATTACHMENT = "attachment"
    COMMENT = "comment"


class SpaceType(str, Enum):
    GLOBAL = "global"
    PERSONAL = "personal"


class ConfluenceObject(BaseModel):
    id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def __init__(self, **data):
        if "id" in data and not isinstance(data["id"], str):
            data["id"] = str(data["id"])

        if "created" in data and data["created"] and "created_at" not in data:
            data["created_at"] = data["created"]

        if "updated" in data and data["updated"] and "updated_at" not in data:
            data["updated_at"] = data["updated"]

        super().__init__(**data)


class ConfluenceSpace(ConfluenceObject):
    key: str
    name: Optional[str] = None
    description: Optional[dict[str, Any]] = None
    type: Optional[SpaceType] = None

    def __init__(self, **data):
        if "name" in data and "title" not in data:
            data["title"] = data["name"]

        if "type" in data and isinstance(data["type"], str):
            try:
                data["type"] = SpaceType(data["type"])
            except ValueError:
                pass

        super().__init__(**data)


class BodyContent(BaseModel):
    view: Optional[dict[str, Any]] = None
    storage: Optional[dict[str, Any]] = None
    plain: Optional[dict[str, Any]] = None

    model_config = {
        "populate_by_name": True,
    }


class Version(BaseModel):
    number: int
    when: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
    }


class ConfluencePage(ConfluenceObject):
    space: Optional[Union[ConfluenceSpace, dict[str, Any]]] = None
    content_type: ContentType = ContentType.PAGE
    body: Optional[BodyContent] = None
    version: Optional[Version] = None
    status: Optional[str] = None

    def __init__(self, **data):
        if "type" in data and "content_type" not in data:
            data["content_type"] = data["type"]

        if "content_type" in data and isinstance(data["content_type"], str):
            try:
                data["content_type"] = ContentType(data["content_type"])
            except ValueError:
                pass

        if "body" in data and isinstance(data["body"], dict):
            data["body"] = BodyContent(**data["body"])

        if "version" in data and isinstance(data["version"], dict):
            data["version"] = Version(**data["version"])

        if "space" in data and isinstance(data["space"], dict):
            pass

        super().__init__(**data)

    @property
    def html_content(self) -> Optional[str]:
        if self.body and self.body.view and "value" in self.body.view:
            return self.body.view["value"]
        return None

    @property
    def storage_content(self) -> Optional[str]:
        if self.body and self.body.storage and "value" in self.body.storage:
            return self.body.storage["value"]
        return None

    @property
    def plain_content(self) -> Optional[str]:
        if self.body and self.body.plain and "value" in self.body.plain:
            return self.body.plain["value"]
        return None


class SearchResult(BaseModel):
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
        if "totalSize" in data:
            data["total_size"] = data["totalSize"]
        elif "total" in data:
            data["total_size"] = data["total"]
        elif "size" in data and "total_size" not in data:
            data["total_size"] = data["size"]

        if "results" in data and isinstance(data["results"], list):
            transformed_results = []
            for item in data["results"]:
                if isinstance(item, dict):
                    transformed_results.append(ConfluencePage(**item))
                elif isinstance(item, ConfluencePage):
                    transformed_results.append(item)
            data["results"] = transformed_results

        super().__init__(**data)
