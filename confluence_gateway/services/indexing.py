import logging
import random
from typing import Any, Optional

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.vector_db import (
    Document,
    VectorDBAdapter,
    get_vector_db_adapter,
)
from confluence_gateway.core.config import vector_db_config

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service responsible for fetching content, processing it (chunking, embedding),
    and storing it in the configured vector database.
    """

    def __init__(self, confluence_client: ConfluenceClient):
        self.confluence_client = confluence_client
        self.vector_db_adapter: Optional[VectorDBAdapter] = get_vector_db_adapter()
        self.vector_db_config = vector_db_config

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

    def _simulate_chunking(self, text: str, chunk_size: int = 200) -> list[str]:
        """Placeholder for actual text chunking logic."""
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _simulate_embedding(self, text_chunk: str) -> Optional[list[float]]:
        """Placeholder for actual embedding generation."""
        if not self.vector_db_config or not self.vector_db_config.embedding_dimension:
            logger.error(
                "Cannot generate dummy embedding: Embedding dimension not configured."
            )
            return None
        # Generate random vector of the configured dimension
        dimension = self.vector_db_config.embedding_dimension
        return [random.random() for _ in range(dimension)]

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

        if not self.vector_db_config or not self.vector_db_config.embedding_dimension:
            logger.error(
                f"Vector DB embedding dimension not configured. Skipping indexing for content ID: {content_id}"
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

        documents_to_upsert: list[Document] = []
        for i, chunk in enumerate(chunks):
            # 2. Simulate Embedding
            embedding = self._simulate_embedding(chunk)
            if embedding is None:
                logger.error(
                    f"Failed to generate embedding for chunk {i} of content {content_id}. Skipping chunk."
                )
                continue

            # 3. Create Document object
            chunk_id = f"{content_id}_chunk_{i}"
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_sequence_number"] = i

            doc = Document(
                id=chunk_id,
                text=chunk,
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
            logger.info(
                f"Upserting {len(documents_to_upsert)} documents for content ID: {content_id} using {self.vector_db_config.type} adapter."
            )
            self.vector_db_adapter.upsert(documents=documents_to_upsert)
            logger.info(f"Successfully upserted documents for content ID: {content_id}")
        except Exception as e:
            logger.error(
                f"Failed to upsert documents for content ID {content_id}: {e}",
                exc_info=True,
            )
