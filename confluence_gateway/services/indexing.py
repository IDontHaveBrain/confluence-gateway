import logging
from typing import Any, Optional

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ConfluenceSpace,
)
from confluence_gateway.adapters.vector_db import (
    Document,
    VectorDBAdapter,
)
from confluence_gateway.core.config import (
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
)
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceConnectionError,
)
from confluence_gateway.services.embedding import EmbeddingError, EmbeddingService

logger = logging.getLogger(__name__)

PAGE_FETCH_LIMIT = 50


class IndexingService:
    def __init__(
        self,
        confluence_client: ConfluenceClient,
        indexing_config: IndexingConfig,
        search_config: SearchConfig,
        vector_db_config: Optional[VectorDBConfig],
        embedding_service: Optional[EmbeddingService] = None,
    ):
        from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter

        self.confluence_client = confluence_client
        self.indexing_config = indexing_config
        self.search_config = search_config
        self.vector_db_config = vector_db_config
        self.embedding_service = embedding_service
        self.vector_db_adapter: Optional[VectorDBAdapter] = get_vector_db_adapter()

        if self.vector_db_adapter:
            if self.vector_db_config:
                logger.info(
                    f"IndexingService initialized with Vector DB Adapter: {self.vector_db_config.type} and Indexing Config: include={self.indexing_config.include_spaces}, exclude={self.indexing_config.exclude_spaces}"
                )
            else:
                logger.warning(
                    "IndexingService initialized with Vector DB Adapter but missing configuration."
                )
        else:
            logger.warning(
                "IndexingService initialized WITHOUT Vector DB Adapter (disabled or config error)."
            )

        if self.embedding_service:
            logger.info("IndexingService initialized with Embedding Service.")
        else:
            logger.warning(
                "IndexingService initialized WITHOUT Embedding Service. Embedding features will be disabled for indexing."
            )

    def _list_all_accessible_spaces(self) -> list[ConfluenceSpace]:
        """Fetches all spaces accessible by the configured credentials."""
        all_spaces = []
        try:
            logger.info("Fetching all accessible spaces from Confluence...")
            # The library's get_all_spaces handles pagination internally
            spaces_data = self.confluence_client.atlassian_api.get_all_spaces(limit=50)
            if spaces_data and "results" in spaces_data:
                for space_dict in spaces_data["results"]:
                    try:
                        space = self.confluence_client._parse_space(space_dict)
                        all_spaces.append(space)
                    except Exception as parse_err:
                        logger.warning(
                            f"Failed to parse space data: {space_dict.get('key', 'N/A')}. Error: {parse_err}"
                        )
                logger.info(f"Successfully fetched {len(all_spaces)} spaces.")
            else:
                logger.warning("No spaces found or unexpected response format.")
        except (ConfluenceAPIError, ConfluenceConnectionError) as e:
            logger.error(f"Failed to fetch spaces from Confluence: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while fetching spaces: {e}",
                exc_info=True,
            )
        return all_spaces

    def _list_target_spaces(self) -> list[ConfluenceSpace]:
        """Lists spaces to be indexed based on configuration."""
        all_spaces = self._list_all_accessible_spaces()
        if not all_spaces:
            return []

        target_spaces = all_spaces
        include_keys_lower = (
            {key.lower() for key in self.indexing_config.include_spaces}
            if self.indexing_config.include_spaces
            else None
        )
        exclude_keys_lower = (
            {key.lower() for key in self.indexing_config.exclude_spaces}
            if self.indexing_config.exclude_spaces
            else set()
        )

        if include_keys_lower:
            target_spaces = [
                space for space in all_spaces if space.key.lower() in include_keys_lower
            ]
            logger.info(
                f"Filtered spaces based on include_spaces config ({len(target_spaces)} remaining)."
            )

        if exclude_keys_lower:
            original_count = len(target_spaces)
            target_spaces = [
                space
                for space in target_spaces
                if space.key.lower() not in exclude_keys_lower
            ]
            if len(target_spaces) < original_count:
                logger.info(
                    f"Filtered spaces based on exclude_spaces config ({len(target_spaces)} remaining)."
                )

        logger.info(
            f"Final list of target spaces for indexing: {[space.key for space in target_spaces]}"
        )
        return target_spaces

    def _list_pages_in_space(self, space_key: str) -> list[ConfluencePage]:
        """Lists all pages within a specific space using paginated CQL search."""
        all_pages: list[ConfluencePage] = []
        start = 0
        limit = PAGE_FETCH_LIMIT
        cql = (
            f'space = "{self.confluence_client._escape_cql(space_key)}" AND type = page'
        )
        logger.info(
            f"Fetching pages for space '{space_key}' using CQL: '{cql}' (limit: {limit})"
        )

        while True:
            try:
                logger.debug(f"Fetching pages for space '{space_key}', start={start}")
                search_result = self.confluence_client.search_by_cql(
                    cql=cql,
                    start=start,
                    limit=limit,
                    get_all_results=False,  # Explicitly manage pagination here
                    expand=["version"],  # Add version for later comparison
                )

                if not search_result or not search_result.results:
                    logger.debug(
                        f"No more pages found for space '{space_key}' at start={start}."
                    )
                    break

                num_fetched = len(search_result.results)
                logger.debug(
                    f"Fetched {num_fetched} pages for space '{space_key}' (total reported: {search_result.total_size})."
                )
                all_pages.extend(search_result.results)

                if start + num_fetched >= search_result.total_size:
                    logger.debug(
                        f"All pages fetched for space '{space_key}' (total: {search_result.total_size})."
                    )
                    break

                start += limit

            except (ConfluenceAPIError, ConfluenceConnectionError) as e:
                logger.error(
                    f"Error fetching pages for space '{space_key}' at start={start}: {e}",
                    exc_info=True,
                )
                break  # Stop fetching for this space on error
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching pages for space '{space_key}' at start={start}: {e}",
                    exc_info=True,
                )
                break  # Stop fetching for this space on error

        logger.info(f"Found {len(all_pages)} pages in space '{space_key}'.")
        return all_pages

    def _simulate_chunking(self, text: str, chunk_size: int = 200) -> list[str]:
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def index_content(
        self,
        content_id: str,
        text_content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.vector_db_adapter:
            logger.warning(
                f"Vector DB Adapter not available. Skipping indexing for content ID: {content_id}"
            )
            return

        if not self.embedding_service:
            logger.warning(
                f"Embedding Service not available. Skipping indexing for content ID: {content_id}"
            )
            return

        try:
            dimension = self.embedding_service.get_dimension()
            if dimension is None:
                logger.error(
                    f"Could not determine embedding dimension from Embedding Service. Skipping indexing for content ID: {content_id}"
                )
                return
        except EmbeddingError as e:
            logger.error(
                f"Error getting embedding dimension: {e}. Skipping indexing for content ID: {content_id}",
                exc_info=True,
            )
            return

        logger.info(f"Starting indexing process for content ID: {content_id}")

        base_metadata = metadata or {}
        base_metadata["original_content_id"] = content_id

        chunks = self._simulate_chunking(text_content)
        if not chunks:
            logger.warning(f"No chunks generated for content ID: {content_id}")
            return

        logger.debug(f"Generated {len(chunks)} chunks for content ID: {content_id}")

        chunk_texts = [chunk for chunk in chunks]

        try:
            logger.info(
                f"Generating embeddings for {len(chunk_texts)} chunks for content ID: {content_id}..."
            )
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")

            if len(embeddings) != len(chunk_texts):
                logger.error(
                    f"Mismatch between number of chunks ({len(chunk_texts)}) and generated embeddings ({len(embeddings)}) for content ID: {content_id}. Skipping upsert."
                )
                return

        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embeddings for content ID {content_id}: {e}",
                exc_info=True,
            )
            return
        except Exception as e:
            logger.error(
                f"Unexpected error during embedding generation for content ID {content_id}: {e}",
                exc_info=True,
            )
            return

        documents_to_upsert: list[Document] = []
        for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            if embedding is None or not embedding:
                logger.warning(
                    f"Skipping chunk {i} for content {content_id} due to missing/empty embedding."
                )
                continue

            chunk_id = f"{content_id}_chunk_{i}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_sequence_number"] = i

            doc = Document(
                id=chunk_id,
                text=chunk_text,
                embedding=embedding,
                metadata=chunk_metadata,
            )
            documents_to_upsert.append(doc)

        if not documents_to_upsert:
            logger.warning(
                f"No valid documents prepared for upsert for content ID: {content_id}"
            )
            return

        try:
            assert self.vector_db_config is not None
            logger.info(
                f"Upserting {len(documents_to_upsert)} documents with real embeddings for content ID: {content_id} using {self.vector_db_config.type} adapter."
            )
            self.vector_db_adapter.upsert(documents=documents_to_upsert)
            logger.info(f"Successfully upserted documents for content ID: {content_id}")
        except Exception as e:
            logger.error(
                f"Failed to upsert documents for content ID {content_id}: {e}",
                exc_info=True,
            )
