"""
EmbeddingManager for handling embedding operations in indexing.

Streamlined version focused on core functionality:
- Content validation and preprocessing
- Embedding generation and vector DB operations
- Simple error handling and logging
"""

import logging
import uuid

from llama_index.core.node_parser import SentenceSplitter

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
)
from confluence_gateway.adapters.vector_db import (
    Document,
    VectorDBAdapter,
)
from confluence_gateway.core.config import (
    IndexingConfig,
    VectorDBConfig,
)
from confluence_gateway.core.exceptions import (
    EmbeddingCompatibilityError,
    EmbeddingError,
)
from confluence_gateway.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Streamlined embedding manager for content indexing.

    Core responsibilities:
    - Index content with embedding generation
    - Handle text chunking and vector DB operations
    - Provide simple content deletion
    """

    def __init__(
        self,
        *,
        confluence_client: ConfluenceClient,
        vector_db_adapter: VectorDBAdapter,
        embedding_service: EmbeddingService,
        indexing_config: IndexingConfig,
        vector_db_config: VectorDBConfig,
    ):
        """Initialize the EmbeddingManager with required services."""
        self.confluence_client = confluence_client
        self.vector_db_adapter = vector_db_adapter
        self.embedding_service = embedding_service
        self.indexing_config = indexing_config
        self.vector_db_config = vector_db_config

        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=vector_db_config.chunk_size,
            chunk_overlap=vector_db_config.chunk_overlap,
        )

        logger.debug(
            f"EmbeddingManager initialized with chunk_size={vector_db_config.chunk_size}, "
            f"chunk_overlap={vector_db_config.chunk_overlap}"
        )

    def index_content(
        self,
        content_object: ConfluencePage | ConfluenceAttachment,
        text_content: str,
    ) -> bool:
        """Index content by generating embeddings and storing in vector DB."""
        content_id = content_object.id
        document_type = (
            "attachment" if isinstance(content_object, ConfluenceAttachment) else "page"
        )

        # Validate prerequisites and content
        if not self._validate_prerequisites(content_object, text_content):
            return False

        try:
            # Validate embedding compatibility
            self.embedding_service.validate_compatibility_with_vector_db(
                self.vector_db_adapter, operation_type="index"
            )

            # Process content: chunk, embed, and store
            chunks = list(self.text_splitter.split_text(text_content))
            if not chunks:
                logger.warning(
                    f"No chunks generated for {document_type} ID: {content_id}"
                )
                return False

            embeddings = self.embedding_service.embed_texts(chunks)
            if len(embeddings) != len(chunks):
                logger.error(
                    f"Embedding count mismatch for {document_type} {content_id}"
                )
                return False

            # Create and store documents
            documents = self._create_documents(
                content_object, document_type, chunks, embeddings
            )
            self.vector_db_adapter.upsert(documents=documents)

            logger.info(
                f"Successfully indexed {document_type} {content_id} with {len(chunks)} chunks"
            )

            # Store model info (non-critical)
            try:
                self.embedding_service.store_model_info_in_vector_db(
                    self.vector_db_adapter
                )
            except Exception as e:
                logger.warning(f"Failed to store model info: {e}")

            return True

        except (EmbeddingCompatibilityError, EmbeddingError) as e:
            logger.error(f"Embedding failed for {document_type} {content_id}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"Indexing failed for {document_type} {content_id}: {e}", exc_info=True
            )
            return False

    def delete_content(self, content_id: str) -> bool:
        """
        Delete content from vector DB.

        Args:
            content_id: ID of the content to delete

        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.vector_db_adapter:
            logger.error("Vector DB adapter not available for content deletion")
            return False

        try:
            self.vector_db_adapter.delete_by_metadata(
                {"original_content_id": content_id}
            )
            logger.info(f"Successfully deleted content {content_id} from vector DB")
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete content {content_id} from vector DB: {e}",
                exc_info=True,
            )
            return False

    def _validate_prerequisites(
        self,
        content_object: ConfluencePage | ConfluenceAttachment,
        text_content: str,
    ) -> bool:
        """Validate all prerequisites for indexing."""
        content_id = content_object.id
        document_type = (
            "attachment" if isinstance(content_object, ConfluenceAttachment) else "page"
        )

        # Check required services and content
        if not all(
            [
                self.vector_db_adapter,
                self.embedding_service,
                self.text_splitter,
                text_content,
            ]
        ):
            logger.warning(
                f"Missing dependencies or content for {document_type} {content_id}"
            )
            return False

        return True

    def _create_documents(
        self,
        content_object: ConfluencePage | ConfluenceAttachment,
        document_type: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> list[Document]:
        """Create Document objects for vector DB storage."""
        # Get base metadata and model info
        base_fields = self.confluence_client.extract_content_fields(content_object)
        model_info = self.embedding_service.get_model_info()
        model_metadata = model_info.to_metadata_dict() if model_info else {}

        documents = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:
                continue

            # Create metadata
            metadata = {
                "document_type": document_type,
                "original_content_id": base_fields.get("id"),
                "title": base_fields.get("title"),
                "space_key": base_fields.get("space_key"),
                "url": base_fields.get("url"),
                "chunk_sequence_number": i,
            }

            # Add timestamps
            if dt := base_fields.get("updated_at") or base_fields.get("created_at"):
                metadata["last_modified"] = dt.isoformat()

            # Add attachment-specific fields
            if document_type == "attachment":
                metadata.update(
                    {
                        "attachment_filename": base_fields.get("file_name"),
                        "parent_page_id": base_fields.get("parent_id"),
                    }
                )

            # Add model info and clean None values
            metadata.update(model_metadata)
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )

        return documents
