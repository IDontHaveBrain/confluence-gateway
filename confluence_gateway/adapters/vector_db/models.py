from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(description="Unique identifier for the document chunk")
    text: str = Field(description="The actual text content of the chunk")
    embedding: list[float] = Field(description="The vector embedding of the text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (e.g., confluence_page_id, space_key, url)",
    )


class VectorSearchResultItem(BaseModel):
    id: str = Field(description="ID of the retrieved document chunk")
    score: float = Field(description="Similarity score (higher is typically better)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Associated metadata from the stored document"
    )
    text: Optional[str] = Field(
        default=None,
        description="The text content of the chunk, if requested/available",
    )
