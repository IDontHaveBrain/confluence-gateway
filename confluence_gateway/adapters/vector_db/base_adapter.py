from abc import ABC, abstractmethod
from typing import Any

from confluence_gateway.core.config import ModelMetadata, VectorDBConfig

from .models import Document, VectorSearchResultItem


class VectorDBAdapter(ABC):
    @abstractmethod
    def __init__(self, config: "VectorDBConfig") -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResultItem]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def search_by_metadata(
        self,
        filters: dict[str, Any],
        select: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def delete_by_metadata(self, filters: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection including size, configuration, etc."""
        pass

    @abstractmethod
    def store_model_metadata(self, metadata: ModelMetadata) -> None:
        """Store model metadata in the vector database."""
        pass

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata | None:
        """Retrieve model metadata from the vector database."""
        pass

    @abstractmethod
    def list_collections(self) -> list[dict[str, Any]]:
        """List all available collections and their basic information."""
        pass

    @abstractmethod
    def get_collection_metadata(self) -> dict[str, Any] | None:
        """Get metadata for the collection, including embedding model information."""
        pass

    @abstractmethod
    def set_collection_metadata(self, metadata: dict[str, Any]) -> None:
        """Set metadata for the collection, including embedding model information."""
        pass

    @abstractmethod
    def has_data(self) -> bool:
        """Check if the collection has any data."""
        pass
