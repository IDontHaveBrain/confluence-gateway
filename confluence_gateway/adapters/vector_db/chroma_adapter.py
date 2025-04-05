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
    def __init__(self, config: "VectorDBConfig") -> None:
        self.config = config
        self.client: Optional[ClientAPI] = None
        self.collection: Optional[Collection] = None

    def initialize(self) -> None:
        if self.config.chroma_host and self.config.chroma_port:
            self.client = chromadb.HttpClient(
                host=self.config.chroma_host, port=self.config.chroma_port
            )
        elif self.config.chroma_persist_path:
            self.client = chromadb.PersistentClient(
                path=self.config.chroma_persist_path
            )
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=None,
        )

    def upsert(self, documents: list[Document]) -> None:
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.text for doc in documents]

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
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        where = filters if filters else None

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

        search_results = []

        if results["ids"] and results["ids"][0]:
            assert results["distances"] is not None, (
                "Distances missing from query results despite being requested"
            )
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity_score = 1.0 - distance

                result = VectorSearchResultItem(
                    id=results["ids"][0][i],
                    score=similarity_score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    text=results["documents"][0][i] if results["documents"] else None,
                )
                search_results.append(result)

        return search_results

    def delete(self, ids: list[str]) -> None:
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        self.collection.delete(ids=ids)

    def count(self) -> int:
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )

        return self.collection.count()

    def close(self) -> None:
        self.client = None
        self.collection = None
