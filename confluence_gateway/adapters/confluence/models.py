from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def __init__(self, **data: Any) -> None:
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
    name: str | None = None
    description: dict[str, Any] | None = None
    type: SpaceType | None = None

    def __init__(self, **data: Any) -> None:
        if "name" in data and "title" not in data:
            data["title"] = data["name"]

        if "type" in data and isinstance(data["type"], str):
            try:
                data["type"] = SpaceType(data["type"])
            except ValueError:
                pass

        super().__init__(**data)


class BodyContent(BaseModel):
    view: dict[str, Any] | None = None
    storage: dict[str, Any] | None = None
    plain: dict[str, Any] | None = None

    model_config = {
        "populate_by_name": True,
    }


class Version(BaseModel):
    number: int
    when: datetime | None = None

    model_config = {
        "populate_by_name": True,
    }


class ConfluencePage(ConfluenceObject):
    space: ConfluenceSpace | dict[str, Any] | None = None
    content_type: ContentType = ContentType.PAGE
    body: BodyContent | None = None
    version: Version | None = None
    status: str | None = None

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    @field_validator("content_type", mode="before")
    @classmethod
    def normalize_content_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return ContentType(v)
            except ValueError:
                pass
        return v

    @model_validator(mode="before")
    @classmethod
    def map_type_to_content_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            if "type" in data and "content_type" not in data:
                data["content_type"] = data["type"]

            if "body" in data and isinstance(data["body"], dict):
                data["body"] = BodyContent(**data["body"])

            if "version" in data and isinstance(data["version"], dict):
                version_obj = Version(**data["version"])
                data["version"] = version_obj
                if version_obj.when:
                    data["updated_at"] = version_obj.when

        return data

    @property
    def html_content(self) -> str | None:
        if self.body and self.body.view and "value" in self.body.view:
            return str(self.body.view["value"])
        return None

    @property
    def storage_content(self) -> str | None:
        if self.body and self.body.storage and "value" in self.body.storage:
            return str(self.body.storage["value"])
        return None

    @property
    def plain_content(self) -> str | None:
        if self.body and self.body.plain and "value" in self.body.plain:
            return str(self.body.plain["value"])
        return None


class ConfluenceAttachmentLinks(BaseModel):
    download: str | None = None
    webui: str | None = None
    self: str | None = None


class ConfluenceAttachmentExtensions(BaseModel):
    mediaType: str | None = Field(None, alias="media-type")
    fileSize: int | None = Field(None, alias="file-size")
    comment: str | None = None


class ConfluenceAttachment(ConfluenceObject):
    content_type: ContentType = ContentType.ATTACHMENT
    status: str | None = None
    extensions: ConfluenceAttachmentExtensions | None = None
    _links: ConfluenceAttachmentLinks | None = None
    version: Version | None = None

    def __init__(self, **data: Any) -> None:
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
    def download_url(self) -> str | None:
        return self._links.download if self._links else None

    @property
    def media_type(self) -> str | None:
        return self.extensions.mediaType if self.extensions else None

    @property
    def file_size(self) -> int | None:
        return self.extensions.fileSize if self.extensions else None


class SearchResult(BaseModel):
    total_size: int = 0
    start: int = 0
    limit: int = 0
    results: list[ConfluencePage] = Field(default_factory=list)

    model_config = {
        "populate_by_name": True,
    }

    def __init__(self, **data: Any) -> None:
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
                elif isinstance(item_data, ConfluencePage | ConfluenceAttachment):
                    transformed_results.append(item_data)
            data["results"] = transformed_results

        super().__init__(**data)
