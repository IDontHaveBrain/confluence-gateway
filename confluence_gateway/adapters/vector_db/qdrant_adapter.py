import logging
from typing import Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig

logger = logging.getLogger(__name__)


class QdrantAdapter(VectorDBAdapter):
    def __init__(self, config: "VectorDBConfig") -> None:
        self.config = config
        self.client: Optional[QdrantClient] = None
        logger.info(f"Initializing QdrantAdapter with config: {config.type}")

    def initialize(self) -> None:
        if not self.config.embedding_dimension:
            raise ValueError(
                "Qdrant adapter requires VECTOR_DB_EMBEDDING_DIMENSION to be set."
            )

        try:
            # Determine the right parameter (url or location) based on qdrant_url value
            client_url = None
            client_location = None

            if self.config.qdrant_url == ":memory:":
                client_location = ":memory:"
                logger.info(
                    f"Initializing Qdrant in memory mode, "
                    f"gRPC Port: {self.config.qdrant_grpc_port}, "
                    f"Prefer gRPC: {self.config.qdrant_prefer_grpc}, "
                    f"API Key Provided: {'Yes' if self.config.qdrant_api_key else 'No'}"
                )
            else:
                client_url = (
                    str(self.config.qdrant_url) if self.config.qdrant_url else None
                )
                logger.info(
                    f"Connecting to Qdrant at URL: {client_url}, "
                    f"gRPC Port: {self.config.qdrant_grpc_port}, "
                    f"Prefer gRPC: {self.config.qdrant_prefer_grpc}, "
                    f"API Key Provided: {'Yes' if self.config.qdrant_api_key else 'No'}"
                )

            self.client = QdrantClient(
                url=client_url,
                location=client_location,
                api_key=self.config.qdrant_api_key,
                grpc_port=self.config.qdrant_grpc_port,
                prefer_grpc=self.config.qdrant_prefer_grpc,
            )

            collection_name = self.config.collection_name
            logger.info(f"Checking for Qdrant collection: {collection_name}")

            collection_exists = False
            try:
                collections_response = self.client.get_collections()
                collection_exists = any(
                    col.name == collection_name
                    for col in collections_response.collections
                )
                logger.debug(f"Collection exists check result: {collection_exists}")
            except UnexpectedResponse as e:
                logger.error(
                    f"Error checking collections in Qdrant: {e}", exc_info=True
                )
                raise ConnectionError(
                    f"Failed to interact with Qdrant collections: {e}"
                ) from e
            except Exception as e:
                logger.error(
                    f"Failed to connect or check collections in Qdrant: {e}",
                    exc_info=True,
                )
                raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

            if not collection_exists:
                logger.info(f"Collection '{collection_name}' not found. Creating...")
                vector_params = models.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=models.Distance.COSINE,
                )
                self.client.create_collection(
                    collection_name=collection_name, vectors_config=vector_params
                )
                logger.info(
                    f"Successfully created collection '{collection_name}' "
                    f"with dimension {self.config.embedding_dimension} and distance {vector_params.distance}."
                )
            else:
                logger.info(f"Using existing Qdrant collection: {collection_name}")

        except (ValueError, ConnectionError) as e:
            logger.error(f"Qdrant initialization failed: {e}", exc_info=True)
            self.client = None
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during Qdrant initialization: {e}", exc_info=True
            )
            self.client = None
            raise RuntimeError(f"Unexpected Qdrant initialization error: {e}") from e

    def _ensure_client(self) -> QdrantClient:
        if not self.client:
            raise RuntimeError(
                "Qdrant client not initialized. Call initialize() first."
            )
        return self.client

    def upsert(self, documents: list[Document]) -> None:
        client = self._ensure_client()
        collection_name = self.config.collection_name

        points_to_upsert = []
        for doc in documents:
            payload = {**doc.metadata, "text": doc.text}
            points_to_upsert.append(
                models.PointStruct(id=doc.id, vector=doc.embedding, payload=payload)
            )

        if not points_to_upsert:
            logger.warning("Upsert called with empty document list.")
            return

        try:
            logger.info(
                f"Upserting {len(points_to_upsert)} points to Qdrant collection '{collection_name}'"
            )
            client.upsert(
                collection_name=collection_name,
                points=points_to_upsert,
                wait=True,
            )
            logger.info(f"Successfully upserted {len(points_to_upsert)} points.")
        except Exception as e:
            logger.error(f"Qdrant upsert operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e

    def _translate_filters(
        self, filters: Optional[dict[str, Any]]
    ) -> Optional[models.Filter]:
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            condition = models.FieldCondition(
                key=key, match=models.MatchValue(value=value)
            )
            must_conditions.append(condition)

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResultItem]:
        client = self._ensure_client()
        collection_name = self.config.collection_name
        qdrant_filter = self._translate_filters(filters)

        try:
            logger.info(
                f"Searching Qdrant collection '{collection_name}' with top_k={top_k}, filters provided: {bool(filters)}"
            )
            query_response = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            logger.info(f"Qdrant query returned {len(query_response.points)} results.")

            output_results = []
            for scored_point in query_response.points:
                payload = scored_point.payload or {}
                text_content = payload.pop("text", None)
                metadata = payload

                output_results.append(
                    VectorSearchResultItem(
                        id=str(scored_point.id),
                        score=scored_point.score,
                        metadata=metadata,
                        text=text_content,
                    )
                )
            return output_results

        except Exception as e:
            logger.error(f"Qdrant query operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant query failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            logger.warning("Delete called with empty ID list.")
            return

        client = self._ensure_client()
        collection_name = self.config.collection_name

        try:
            logger.info(
                f"Deleting {len(ids)} points from Qdrant collection '{collection_name}'"
            )
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            logger.info(f"Successfully deleted {len(ids)} points.")
        except Exception as e:
            logger.error(f"Qdrant delete operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant delete failed: {e}") from e

    def count(self) -> int:
        client = self._ensure_client()
        collection_name = self.config.collection_name

        try:
            logger.info(f"Counting points in Qdrant collection '{collection_name}'")
            count_result = client.count(
                collection_name=collection_name,
                exact=True,
            )
            logger.info(f"Qdrant count result: {count_result.count}")
            return count_result.count
        except Exception as e:
            logger.error(f"Qdrant count operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant count failed: {e}") from e

    def close(self) -> None:
        if self.client:
            try:
                logger.info("Closing Qdrant client connection.")
                self.client.close()
                logger.info("Qdrant client closed.")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}", exc_info=True)
            finally:
                self.client = None
        else:
            logger.debug("Qdrant client already closed or was never initialized.")
