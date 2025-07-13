import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig

# Lazy loading for ChromaDB dependencies
_chromadb = None
_chroma_client_api = None
_chroma_collection = None
_chroma_types = None


def _get_chroma_deps() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Lazy load ChromaDB dependencies with caching."""
    global _chromadb, _chroma_client_api, _chroma_collection, _chroma_types
    if _chromadb is None:
        try:
            import chromadb
            from chromadb.api import ClientAPI
            from chromadb.api.models.Collection import Collection
            from chromadb.api.types import Metadata, Metadatas, Where

            _chromadb = chromadb
            _chroma_client_api = ClientAPI
            _chroma_collection = Collection
            _chroma_types = {
                "Metadata": Metadata,
                "Metadatas": Metadatas,
                "Where": Where,
            }
        except ImportError as e:
            raise ImportError(
                "chromadb is required for ChromaDB vector database. "
                "Install it with: pip install chromadb"
            ) from e
    return _chromadb, _chroma_client_api, _chroma_collection, _chroma_types


logger = logging.getLogger(__name__)


class ChromaDBAdapter(VectorDBAdapter):
    def __init__(self, config: "VectorDBConfig") -> None:
        self.config = config
        self.client: Any | None = None  # Will be ClientAPI after lazy loading
        self.collection: Any | None = None  # Will be Collection after lazy loading
        logger.info(f"Initializing ChromaDBAdapter with config: {config.type}")

    def initialize(self) -> None:
        # Lightweight configuration validation only - no network calls
        try:
            # Validate dependencies are available
            _get_chroma_deps()
        except ImportError as e:
            logger.error(f"ChromaDB dependencies not available: {e}", exc_info=True)
            raise

        logger.info("ChromaDBAdapter initialized with config validation complete")

    def _ensure_collection(self) -> Any:  # Returns Collection after lazy loading
        if self.collection is not None:
            return self.collection

        try:
            logger.info("Establishing ChromaDB connection...")
            chromadb, _, _, _ = _get_chroma_deps()

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

                logger.info(f"Using PersistentClient: path={persist_path_str}")
                self.client = chromadb.PersistentClient(path=persist_path_str)
            else:
                logger.info("Using transient in-memory Client.")
                self.client = chromadb.Client()

            collection_name = self.config.collection_name
            logger.info(f"Getting or creating ChromaDB collection: {collection_name}")
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(
                f"Successfully established ChromaDB connection and collection '{collection_name}'."
            )

        except Exception as e:
            logger.error(
                f"ChromaDB connection establishment failed: {e}", exc_info=True
            )
            self.client = None
            self.collection = None
            raise RuntimeError(f"ChromaDB connection establishment failed: {e}") from e

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
                metadatas=metadatas,  # Type checking will be done at runtime by ChromaDB
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
        chroma_where_filter = (
            filters  # Type checking will be done at runtime by ChromaDB
        )

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

                metadatas_list: list[
                    Any
                ] = []  # Will be list[Metadata] after lazy loading
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
                    # ChromaDB distances can be > 1.0, so we need to normalize
                    # Convert distance to similarity score (0 to 1, where 1 is most similar)
                    if distance <= 0:
                        similarity_score = 1.0
                    elif distance >= 2.0:
                        similarity_score = 0.0
                    else:
                        # Normalize: distance 0->1, distance 2->0
                        similarity_score = max(0.0, 1.0 - (float(distance) / 2.0))

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
        chroma_where_filter = (
            filters  # Type checking will be done at runtime by ChromaDB
        )

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
        chroma_where_filter = (
            filters  # Type checking will be done at runtime by ChromaDB
        )

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
