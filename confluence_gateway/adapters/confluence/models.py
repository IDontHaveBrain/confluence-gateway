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

        if "updated" in data and data["updated"]:
            data["updated_at"] = data["updated"]
        elif "created" in data and data["created"]:
            if "updated_at" not in data:
                data["updated_at"] = data["created"]

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
            version_obj = Version(**data["version"])
            data["version"] = version_obj
            if version_obj.when:
                data["updated_at"] = version_obj.when

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


class ConfluenceAttachmentLinks(BaseModel):
    download: Optional[str] = None
    webui: Optional[str] = None
    self: Optional[str] = None


class ConfluenceAttachmentExtensions(BaseModel):
    mediaType: Optional[str] = Field(None, alias="media-type")
    fileSize: Optional[int] = Field(None, alias="file-size")
    comment: Optional[str] = None


class ConfluenceAttachment(ConfluenceObject):
    content_type: ContentType = ContentType.ATTACHMENT
    status: Optional[str] = None
    extensions: Optional[ConfluenceAttachmentExtensions] = None
    _links: Optional[ConfluenceAttachmentLinks] = None
    version: Optional[Version] = None

    def __init__(self, **data):
        if "type" in data and "content_type" not in data:
            data["content_type"] = data["type"]

        if "content_type" in data and isinstance(data["content_type"], str):
            try:
                data["content_type"] = ContentType(data["content_type"])
            except ValueError:
                pass

        if "_links" in data and isinstance(data["_links"], dict):
            data["_links"] = ConfluenceAttachmentLinks(**data["_links"])

        if "extensions" in data and isinstance(data["extensions"], dict):
            data["extensions"] = ConfluenceAttachmentExtensions(**data["extensions"])

        if "version" in data and isinstance(data["version"], dict):
            version_obj = Version(**data["version"])
            data["version"] = version_obj
            if version_obj.when:
                data["updated_at"] = version_obj.when

        super().__init__(**data)

    @property
    def download_url(self) -> Optional[str]:
        return self._links.download if self._links else None

    @property
    def media_type(self) -> Optional[str]:
        return self.extensions.mediaType if self.extensions else None

    @property
    def file_size(self) -> Optional[int]:
        return self.extensions.fileSize if self.extensions else None


class SearchResult(BaseModel):
    total_size: int = 0
    start: int = 0
    limit: int = 0
    results: list[ConfluencePage] = Field(default_factory=list)

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
            for item_data in data["results"]:
                if isinstance(item_data, dict):
                    item_type = item_data.get("type")
                    if not item_type and "content" in item_data:
                        content_data = item_data.get("content", {})
                        item_type = content_data.get("type")
                        merged_data = item_data.copy()
                        merged_data.update(content_data)
                        item_data = merged_data

                    if item_type == ContentType.ATTACHMENT.value:
                        transformed_results.append(ConfluenceAttachment(**item_data))
                    else:
                        transformed_results.append(ConfluencePage(**item_data))
                elif isinstance(item_data, (ConfluencePage, ConfluenceAttachment)):
                    transformed_results.append(item_data)
            data["results"] = transformed_results

        super().__init__(**data)
