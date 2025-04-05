import logging
from typing import Any, Optional

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.vector_db import (
    Document,
    VectorDBAdapter,
)

# Import get_vector_db_adapter locally in __init__ to break cycle
# from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.core.config import vector_db_config
from confluence_gateway.services.embedding import EmbeddingError, EmbeddingService

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service responsible for fetching content, processing it (chunking, embedding),
    and storing it in the configured vector database.
    """

    def __init__(
        self,
        confluence_client: ConfluenceClient,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        # Import locally to break potential import cycles
        from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter

        self.confluence_client = confluence_client
        self.vector_db_adapter: Optional[VectorDBAdapter] = get_vector_db_adapter()
        self.vector_db_config = vector_db_config
        self.embedding_service = embedding_service

        if self.vector_db_adapter:
            if self.vector_db_config:
                logger.info(
                    f"IndexingService initialized with Vector DB Adapter: {self.vector_db_config.type}"
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

    def _simulate_chunking(self, text: str, chunk_size: int = 200) -> list[str]:
        """Placeholder for actual text chunking logic."""
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def index_content(
        self,
        content_id: str,
        text_content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Simulates indexing content into the vector database.

        Args:
            content_id: Unique ID of the content (e.g., Confluence page ID).
            text_content: The textual content to be indexed.
            metadata: Additional metadata associated with the content.
        """
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

        # 1. Simulate Chunking
        chunks = self._simulate_chunking(text_content)
        if not chunks:
            logger.warning(f"No chunks generated for content ID: {content_id}")
            return

        logger.debug(f"Generated {len(chunks)} chunks for content ID: {content_id}")

        # Prepare for batch embedding
        chunk_texts = [chunk for chunk in chunks]

        # 2. Generate Embeddings
        try:
            logger.info(
                f"Generating embeddings for {len(chunk_texts)} chunks for content ID: {content_id}..."
            )
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")

            # Add a check for mismatched counts
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

        # 3. Create Document objects
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

        # 4. Upsert Documents using the adapter
        try:
            # Ensure vector_db_config is not None for mypy
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
