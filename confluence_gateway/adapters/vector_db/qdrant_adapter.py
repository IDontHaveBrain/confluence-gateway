import logging
from pathlib import Path
from typing import Any

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig

logger = logging.getLogger(__name__)

# Lazy loading for Qdrant dependencies
_qdrant_client = None
_qdrant_models = None
_qdrant_exceptions = None
_qdrant_filter = None
_qdrant_update_result = None
_qdrant_payload_selector = None


def _get_qdrant_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Lazy load Qdrant dependencies with caching."""
    global \
        _qdrant_client, \
        _qdrant_models, \
        _qdrant_exceptions, \
        _qdrant_filter, \
        _qdrant_update_result, \
        _qdrant_payload_selector

    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient, models
            from qdrant_client.http.exceptions import UnexpectedResponse
            from qdrant_client.models import Filter as QdrantFilter
            from qdrant_client.models import PayloadSelector, UpdateResult

            _qdrant_client = QdrantClient
            _qdrant_models = models
            _qdrant_exceptions = UnexpectedResponse
            _qdrant_filter = QdrantFilter
            _qdrant_update_result = UpdateResult
            _qdrant_payload_selector = PayloadSelector
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for Qdrant vector database. "
                "Install it with: uv add qdrant-client"
            ) from e

    return (
        _qdrant_client,
        _qdrant_models,
        _qdrant_exceptions,
        _qdrant_filter,
        _qdrant_update_result,
        _qdrant_payload_selector,
    )


class QdrantAdapter(VectorDBAdapter):
    def __init__(self, config: VectorDBConfig) -> None:
        self.config = config
        self.client: Any | None = None  # QdrantClient will be loaded lazily
        logger.info(f"Initializing QdrantAdapter with config: {config.type}")

    def initialize(self) -> None:
        # Lightweight configuration validation only - no network calls
        if not self.config.embedding_dimension:
            raise ValueError(
                "Qdrant adapter requires VECTOR_DB_EMBEDDING_DIMENSION to be set."
            )

        # Validate dependencies are available
        try:
            _get_qdrant_deps()
        except ImportError as e:
            logger.error(f"Qdrant dependencies not available: {e}", exc_info=True)
            raise

        logger.info("QdrantAdapter initialized with config validation complete")

    def _ensure_client(self) -> Any:  # Returns QdrantClient but loaded lazily
        if self.client is not None:
            return self.client

        try:
            # Load Qdrant dependencies
            (
                QdrantClient,
                models,
                UnexpectedResponse,
                QdrantFilter,
                UpdateResult,
                PayloadSelector,
            ) = _get_qdrant_deps()
            client_url = None
            client_location = None

            logger.debug(
                f"Config values: qdrant_url={self.config.qdrant_url!r}, qdrant_local_path={self.config.qdrant_local_path!r}"
            )

            # Check if local path is specified
            if self.config.qdrant_local_path:
                # Expand user home directory if present
                local_path = Path(self.config.qdrant_local_path).expanduser()
                # Create directory if it doesn't exist
                local_path.mkdir(parents=True, exist_ok=True)
                client_location = str(local_path)
                logger.info(
                    f"Establishing Qdrant connection with local storage at: {client_location}, "
                    f"gRPC Port: {self.config.qdrant_grpc_port}, "
                    f"Prefer gRPC: {self.config.qdrant_prefer_grpc}, "
                    f"API Key Provided: {'Yes' if self.config.qdrant_api_key else 'No'}"
                )
            elif self.config.qdrant_url == ":memory:":
                client_location = ":memory:"
                logger.info(
                    f"Establishing Qdrant connection in memory mode, "
                    f"gRPC Port: {self.config.qdrant_grpc_port}, "
                    f"Prefer gRPC: {self.config.qdrant_prefer_grpc}, "
                    f"API Key Provided: {'Yes' if self.config.qdrant_api_key else 'No'}"
                )
                logger.debug(
                    f"Memory mode confirmed: qdrant_url='{self.config.qdrant_url}', type={type(self.config.qdrant_url)}"
                )
            else:
                client_url = (
                    str(self.config.qdrant_url) if self.config.qdrant_url else None
                )
                logger.info(
                    f"Establishing Qdrant connection to URL: {client_url}, "
                    f"gRPC Port: {self.config.qdrant_grpc_port}, "
                    f"Prefer gRPC: {self.config.qdrant_prefer_grpc}, "
                    f"API Key Provided: {'Yes' if self.config.qdrant_api_key else 'No'}"
                )

            logger.debug(
                f"Creating QdrantClient with url={client_url}, location={client_location}"
            )

            # For local mode, use path parameter
            if client_location:
                self.client = QdrantClient(
                    path=client_location,
                )
            else:
                # For server mode, pass url and other parameters
                self.client = QdrantClient(
                    url=client_url,
                    api_key=self.config.qdrant_api_key,
                    grpc_port=self.config.qdrant_grpc_port,
                    prefer_grpc=self.config.qdrant_prefer_grpc,
                )

            # Setup collection on first connection
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
                self.client = None
                raise ConnectionError(
                    f"Failed to interact with Qdrant collections: {e}"
                ) from e
            except Exception as e:
                logger.error(
                    f"Failed to connect or check collections in Qdrant: {e}",
                    exc_info=True,
                )
                self.client = None
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
            logger.error(f"Qdrant connection establishment failed: {e}", exc_info=True)
            self.client = None
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during Qdrant connection establishment: {e}",
                exc_info=True,
            )
            self.client = None
            raise RuntimeError(f"Unexpected Qdrant connection error: {e}") from e

        return self.client

    def upsert(self, documents: list[Document]) -> None:
        client = self._ensure_client()
        collection_name = self.config.collection_name

        # Load Qdrant models
        _, models, _, _, _, _ = _get_qdrant_deps()

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
        self, filters: dict[str, Any] | None
    ) -> Any | None:  # Returns models.Filter but loaded lazily
        if not filters:
            return None

        # Load Qdrant dependencies
        _, models, _, QdrantFilter, _, _ = _get_qdrant_deps()

        must_conditions = []
        for key, value in filters.items():
            condition = models.FieldCondition(
                key=key, match=models.MatchValue(value=value)
            )
            must_conditions.append(condition)

        if not must_conditions:
            return None

        if not must_conditions:
            return None

        return QdrantFilter(must=must_conditions)

    def _build_qdrant_filter(
        self, filters: dict[str, Any] | None
    ) -> Any | None:  # Returns QdrantFilter but loaded lazily
        if not filters:
            return None

        # Load Qdrant dependencies
        _, models, _, QdrantFilter, _, _ = _get_qdrant_deps()

        must_conditions = []
        for key, value in filters.items():
            condition = models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            must_conditions.append(condition)

        if not must_conditions:
            return None

        return QdrantFilter(must=must_conditions)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
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

    def search_by_metadata(
        self,
        filters: dict[str, Any],
        select: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        collection_name = self.config.collection_name
        qdrant_filter = self._build_qdrant_filter(filters)

        if not qdrant_filter:
            logger.warning("search_by_metadata called with empty or invalid filters.")
            return []

        # Load Qdrant dependencies
        _, models, _, _, _, PayloadSelector = _get_qdrant_deps()

        payload_selector: Any | bool | None  # PayloadSelector loaded lazily
        if select:
            payload_selector = models.PayloadSelectorInclude(include=select)
        else:
            payload_selector = True

        results = []
        next_offset = None
        retrieved_count = 0

        try:
            logger.info(
                f"Scrolling Qdrant collection '{collection_name}' with filter: {filters}, select: {select}, limit: {limit}"
            )
            while True:
                scroll_limit = 50
                if limit is not None:
                    remaining = limit - retrieved_count
                    if remaining <= 0:
                        break
                    scroll_limit = min(scroll_limit, remaining)

                points, next_offset_value = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=qdrant_filter,
                    limit=scroll_limit,
                    offset=next_offset,
                    with_payload=payload_selector,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload or {}
                    doc_data = {"id": str(point.id)}
                    if select:
                        for key in select:
                            if key in payload:
                                doc_data[key] = payload[key]
                    else:
                        doc_data.update(
                            {k: v for k, v in payload.items() if k != "text"}
                        )

                    results.append(doc_data)
                    retrieved_count += 1

                next_offset = next_offset_value
                if not next_offset or (limit is not None and retrieved_count >= limit):
                    break

            logger.info(f"Qdrant scroll returned {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Qdrant scroll operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant metadata search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            logger.warning("Delete called with empty ID list.")
            return

        client = self._ensure_client()
        collection_name = self.config.collection_name

        # Load Qdrant models
        _, models, _, _, _, _ = _get_qdrant_deps()

        try:
            logger.info(
                f"Deleting {len(ids)} points from Qdrant collection '{collection_name}'"
            )
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            logger.info(f"Successfully deleted {len(ids)} points based on IDs.")
        except Exception as e:
            logger.error(f"Qdrant delete by ID operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant delete by ID failed: {e}") from e

    def delete_by_metadata(self, filters: dict[str, Any]) -> None:
        if not filters:
            logger.warning("delete_by_metadata called with empty filters.")
            return

        client = self._ensure_client()
        collection_name = self.config.collection_name
        qdrant_filter = self._build_qdrant_filter(filters)

        if not qdrant_filter:
            logger.warning("delete_by_metadata filter construction failed.")
            return

        # Load Qdrant dependencies
        _, models, _, _, UpdateResult, _ = _get_qdrant_deps()

        try:
            logger.info(
                f"Deleting points from Qdrant collection '{collection_name}' matching filter: {filters}"
            )
            result: Any = client.delete(  # UpdateResult loaded lazily
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True,
            )
            if result.status == models.UpdateStatus.COMPLETED:
                logger.info(
                    f"Qdrant delete by metadata operation completed for filter: {filters}"
                )
            else:
                logger.warning(
                    f"Qdrant delete by metadata operation status: {result.status} for filter: {filters}"
                )

        except Exception as e:
            logger.error(
                f"Qdrant delete by metadata operation failed: {e}", exc_info=True
            )
            raise RuntimeError(f"Qdrant delete by metadata failed: {e}") from e

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
            return int(count_result.count)
        except Exception as e:
            logger.error(f"Qdrant count operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant count failed: {e}") from e

    def retrieve_by_ids(
        self, ids: list[str], with_payload: bool = True, with_vector: bool = False
    ) -> list[Any]:  # Returns list[models.Record] but loaded lazily
        client = self._ensure_client()
        collection_name = self.config.collection_name
        if not ids:
            logger.warning("retrieve_by_ids called with empty ID list.")
            return []
        try:
            logger.info(
                f"Retrieving {len(ids)} points by ID from Qdrant collection '{collection_name}'"
            )
            records = client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=with_payload,
                with_vectors=with_vector,
            )
            logger.info(f"Qdrant retrieve returned {len(records)} records.")
            return list(records)
        except Exception as e:
            logger.error(f"Qdrant retrieve by ID operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant retrieve by ID failed: {e}") from e

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
