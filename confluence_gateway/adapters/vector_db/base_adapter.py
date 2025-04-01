from abc import ABC, abstractmethod
from typing import Any, Optional

from confluence_gateway.core.config import VectorDBConfig

from .models import Document, VectorSearchResultItem


class VectorDBAdapter(ABC):
    """
    Abstract interface for vector database operations.

    This class defines the common operations that all vector database
    implementations must support, allowing the application to switch
    between different vector databases through configuration.
    """

    @abstractmethod
    def __init__(self, config: "VectorDBConfig") -> None:
        """
        Initialize the adapter with configuration.

        Args:
            config: Vector database configuration containing connection details
                   and other settings specific to the implementation.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Set up the vector database connection and ensure collections/tables exist.

        This method should handle connection establishment, collection/table creation,
        and any other setup required before the adapter can be used.

        Raises:
            Exception: If initialization fails (connection error, permission issues, etc.)
        """
        pass

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None:
        """
        Add or update documents in the vector database.

        Args:
            documents: List of Document objects containing text, embeddings, and metadata
                      to be stored or updated in the vector database.

        Raises:
            Exception: If the operation fails
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResultItem]:
        """
        Search for similar vectors in the database.

        Args:
            query_embedding: Vector representation of the query
            top_k: Number of most similar results to return
            filters: Optional metadata filters to apply to the search

        Returns:
            List of search results with similarity scores and metadata

        Raises:
            Exception: If the search operation fails
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the vector database by their IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            Exception: If the delete operation fails
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of documents in the vector database.

        Returns:
            Count of documents/vectors in the collection

        Raises:
            Exception: If the count operation fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources used by the adapter.

        This method should close connections and free any resources.
        For some implementations, this might be a no-op if resources
        are managed automatically.

        Raises:
            Exception: If cleanup fails
        """
        pass
