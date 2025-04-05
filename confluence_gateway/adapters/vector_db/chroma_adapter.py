from collections.abc import Sequence
from typing import Any, Optional, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import IncludeEnum, Metadatas

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig


class ChromaDBAdapter(VectorDBAdapter):
    """
    ChromaDB implementation of the VectorDBAdapter interface.

    This adapter supports both persistent local storage and client/server modes
    depending on the configuration provided.
    """

    def __init__(self, config: "VectorDBConfig") -> None:
        self.config = config
        self.client: Optional[ClientAPI] = None
        self.collection: Optional[Collection] = None

    def initialize(self) -> None:
        """
        Initialize the ChromaDB client and collection.

        Creates a client based on configuration (persistent or HTTP)
        and gets or creates the collection.

        Raises:
            Exception: If initialization fails
        """
        # Determine client type based on configuration
        if self.config.chroma_host and self.config.chroma_port:
            # Client/server mode takes precedence
            self.client = chromadb.HttpClient(
                host=self.config.chroma_host, port=self.config.chroma_port
            )
        elif self.config.chroma_persist_path:
            # Persistent mode
            self.client = chromadb.PersistentClient(
                path=self.config.chroma_persist_path
            )
        else:
            # In-memory mode (for testing/development)
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=None,  # We provide pre-computed embeddings
        )

    def upsert(self, documents: list[Document]) -> None:
        """
        Add or update documents in ChromaDB.

        Args:
            documents: List of Document objects to upsert

        Raises:
            Exception: If the upsert operation fails
        """
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        # Transform documents to ChromaDB format
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.text for doc in documents]

        # Perform upsert
        self.collection.upsert(
            ids=ids,
            embeddings=cast(list[Sequence[float]], embeddings),
            metadatas=cast(Optional[Metadatas], metadatas),
            documents=texts,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResultItem]:
        """
        Search for similar vectors in ChromaDB.

        Args:
            query_embedding: Vector representation of the query
            top_k: Number of most similar results to return
            filters: Optional metadata filters to apply to the search

        Returns:
            List of search results with similarity scores and metadata

        Raises:
            Exception: If the search operation fails
        """
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        # Prepare where clause for filtering
        where = filters if filters else None

        # Perform search
        results = self.collection.query(
            query_embeddings=cast(list[Sequence[float]], [query_embedding]),
            n_results=top_k,
            where=where,
            include=[
                IncludeEnum.metadatas,
                IncludeEnum.distances,
                IncludeEnum.documents,
            ],
        )

        # Transform results to VectorSearchResultItem format
        search_results = []

        # ChromaDB returns results in a dict with lists
        if results["ids"] and results["ids"][0]:
            # Ensure distances are available since we requested them
            assert results["distances"] is not None, (
                "Distances missing from query results despite being requested"
            )
            for i in range(len(results["ids"][0])):
                # Convert distance to similarity score (1 - distance for L2 distance)
                # For cosine distance, this is already a similarity score
                distance = results["distances"][0][i]
                similarity_score = 1.0 - distance  # Adjust based on distance metric

                result = VectorSearchResultItem(
                    id=results["ids"][0][i],
                    score=similarity_score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    text=results["documents"][0][i] if results["documents"] else None,
                )
                search_results.append(result)

        return search_results

    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from ChromaDB by their IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            Exception: If the delete operation fails
        """
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        self.collection.delete(ids=ids)

    def count(self) -> int:
        """
        Get the total number of documents in ChromaDB.

        Returns:
            Count of documents in the collection

        Raises:
            Exception: If the count operation fails
        """
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        return self.collection.count()

    def close(self) -> None:
        """
        Clean up resources used by the adapter.

        For ChromaDB, this is typically a no-op as the client
        handles connection cleanup automatically.
        """
        # ChromaDB doesn't require explicit cleanup in most cases
        self.client = None
        self.collection = None
