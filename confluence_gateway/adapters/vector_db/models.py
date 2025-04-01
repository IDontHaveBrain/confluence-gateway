from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a document chunk with its embedding and metadata.

    Standard metadata keys:
    - confluence_page_id: ID of the Confluence page
    - confluence_space_key: Key of the Confluence space
    - document_url: URL to the document
    - last_modified_date: When the document was last modified
    - chunk_sequence_number: Position of this chunk in the original document
    - title: Document title
    """

    id: str = Field(description="Unique identifier for the document chunk")
    text: str = Field(description="The actual text content of the chunk")
    embedding: list[float] = Field(description="The vector embedding of the text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (e.g., confluence_page_id, space_key, url)",
    )


class VectorSearchResultItem(BaseModel):
    """
    Represents a search result from the vector database, including similarity score
    and the retrieved document's metadata.
    """

    id: str = Field(description="ID of the retrieved document chunk")
    score: float = Field(description="Similarity score (higher is typically better)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Associated metadata from the stored document"
    )
    text: Optional[str] = Field(
        default=None,
        description="The text content of the chunk, if requested/available",
    )
