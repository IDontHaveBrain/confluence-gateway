from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResultItem(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None
