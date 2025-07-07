import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, Metadatas, Where

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig

logger = logging.getLogger(__name__)


class ChromaDBAdapter(VectorDBAdapter):
    def __init__(self, config: "VectorDBConfig") -> None:
        self.config = config
        self.client: ClientAPI | None = None
        self.collection: Collection | None = None
        logger.info(f"Initializing ChromaDBAdapter with config: {config.type}")

    def initialize(self) -> None:
        try:
            logger.info("Connecting to ChromaDB...")
            if self.config.chroma_host and self.config.chroma_port:
                logger.info(
                    f"Using HttpClient: host={self.config.chroma_host}, port={self.config.chroma_port}"
                )
                self.client = chromadb.HttpClient(
                    host=self.config.chroma_host, port=self.config.chroma_port
                )
            elif self.config.chroma_persist_path:
                # Expand user home directory if present
                persist_path = Path(self.config.chroma_persist_path).expanduser()
                # Create directory if it doesn't exist
                persist_path.mkdir(parents=True, exist_ok=True)
                persist_path_str = str(persist_path)

                logger.info(
                    f"Using PersistentClient: path={persist_path_str}"
                )
                self.client = chromadb.PersistentClient(
                    path=persist_path_str
                )
            else:
                logger.info("Using transient in-memory Client.")
                self.client = chromadb.Client()

            collection_name = self.config.collection_name
            logger.info(f"Getting or creating ChromaDB collection: {collection_name}")
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(
                f"Successfully initialized ChromaDB and collection '{collection_name}'."
            )

        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}", exc_info=True)
            self.client = None
            self.collection = None
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    def _ensure_collection(self) -> Collection:
        if not self.collection:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call initialize() first."
            )
        return self.collection

    def upsert(self, documents: list[Document]) -> None:
        collection = self._ensure_collection()
        if not documents:
            logger.warning("Upsert called with empty document list.")
            return

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.text for doc in documents]

        try:
            logger.info(
                f"Upserting {len(ids)} documents to ChromaDB collection '{collection.name}'"
            )
            collection.upsert(
                ids=ids,
                embeddings=cast(list[Sequence[float]], embeddings),
                metadatas=cast(Metadatas | None, metadatas),
                documents=texts,
            )
            logger.info(f"Successfully upserted {len(ids)} documents.")
        except Exception as e:
            logger.error(f"ChromaDB upsert operation failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB upsert failed: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResultItem]:
        collection = self._ensure_collection()
        chroma_where_filter = cast(Where | None, filters) if filters else None

        try:
            logger.info(
                f"Querying ChromaDB collection '{collection.name}' with top_k={top_k}, filters provided: {bool(filters)}"
            )
            results = collection.query(
                query_embeddings=cast(list[Sequence[float]], [query_embedding]),
                n_results=top_k,
                where=chroma_where_filter,
                include=[
                    "metadatas",
                    "distances",
                    "documents",
                ],
            )
            logger.info(
                f"ChromaDB query returned results for {len(results.get('ids', [[]])[0])} items."
            )

            search_results = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]

                distances: list[float] = []
                distances_outer = results.get("distances")
                if distances_outer is not None and len(distances_outer) > 0:
                    distances = distances_outer[0]
                else:
                    logger.warning(
                        "Distances missing from ChromaDB query results despite being requested"
                    )
                    distances = [1.0] * len(ids)

                metadatas_list: list[Metadata] = []
                metadatas_outer = results.get("metadatas")
                if metadatas_outer is not None and len(metadatas_outer) > 0:
                    metadatas_list = metadatas_outer[0]
                else:
                    logger.warning(
                        "Metadatas missing from ChromaDB query results despite being requested"
                    )
                    metadatas_list = [{}] * len(ids)

                documents_list: list[str | None] = [None] * len(ids)
                documents_outer = results.get("documents")
                if documents_outer is not None and len(documents_outer) > 0:
                    documents_list = cast(list[str | None], documents_outer[0])

                for i, item_id in enumerate(ids):
                    metadata = metadatas_list[i] if i < len(metadatas_list) else {}
                    text_content = (
                        documents_list[i] if i < len(documents_list) else None
                    )

                    distance = (
                        distances[i]
                        if i < len(distances) and distances[i] is not None
                        else 1.0
                    )
                    similarity_score = 1.0 - float(distance)

                    result = VectorSearchResultItem(
                        id=item_id,
                        score=similarity_score,
                        metadata=metadata,
                        text=text_content,
                    )
                    search_results.append(result)

            return search_results
        except Exception as e:
            logger.error(f"ChromaDB query operation failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB query failed: {e}") from e

    def search_by_metadata(
        self,
        filters: dict[str, Any],
        select: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        collection = self._ensure_collection()
        chroma_where_filter = cast(Where | None, filters) if filters else None

        if not chroma_where_filter:
            logger.warning("search_by_metadata called with empty filters.")
            return []

        try:
            logger.info(
                f"Getting documents from ChromaDB collection '{collection.name}' with filter: {filters}, select: {select}, limit: {limit}"
            )
            get_results = collection.get(
                where=chroma_where_filter,
                limit=limit,
                include=["metadatas"],
            )

            output_results = []
            if get_results and get_results.get("ids"):
                ids = get_results["ids"]
                metadatas_list = get_results.get("metadatas") or [{}] * len(ids)

                for i, item_id in enumerate(ids):
                    full_metadata = metadatas_list[i] if i < len(metadatas_list) else {}
                    doc_data: dict[str, Any] = {"id": item_id}
                    if select:
                        for key in select:
                            if key in full_metadata:
                                doc_data[key] = full_metadata[key]
                    else:
                        doc_data.update(
                            {k: v for k, v in full_metadata.items() if k != "text"}
                        )

                    output_results.append(doc_data)

            logger.info(f"ChromaDB get returned {len(output_results)} results.")
            return output_results

        except Exception as e:
            logger.error(f"ChromaDB get operation failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB metadata search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        collection = self._ensure_collection()
        if not ids:
            logger.warning("Delete called with empty ID list.")
            return

        try:
            logger.info(
                f"Deleting {len(ids)} documents by ID from ChromaDB collection '{collection.name}'"
            )
            collection.delete(ids=ids)
            logger.info(
                f"Successfully submitted deletion request for {len(ids)} documents by ID."
            )
        except Exception as e:
            logger.error(f"ChromaDB delete by ID operation failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB delete by ID failed: {e}") from e

    def delete_by_metadata(self, filters: dict[str, Any]) -> None:
        collection = self._ensure_collection()
        chroma_where_filter = cast(Where | None, filters) if filters else None

        if not chroma_where_filter:
            logger.warning("delete_by_metadata called with empty filters.")
            return

        try:
            logger.info(
                f"Deleting documents from ChromaDB collection '{collection.name}' matching filter: {filters}"
            )
            collection.delete(where=chroma_where_filter)
            logger.info(
                f"Successfully submitted deletion request for documents matching filter: {filters}"
            )
        except Exception as e:
            logger.error(
                f"ChromaDB delete by metadata operation failed: {e}", exc_info=True
            )
            raise RuntimeError(f"ChromaDB delete by metadata failed: {e}") from e

    def count(self) -> int:
        collection = self._ensure_collection()
        try:
            logger.info(
                f"Counting documents in ChromaDB collection '{collection.name}'"
            )
            count_result = collection.count()
            logger.info(f"ChromaDB count result: {count_result}")
            return int(count_result)
        except Exception as e:
            logger.error(f"ChromaDB count operation failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB count failed: {e}") from e

    def close(self) -> None:
        logger.info(
            "Closing ChromaDBAdapter (releasing client and collection references)."
        )
        self.client = None
        self.collection = None
