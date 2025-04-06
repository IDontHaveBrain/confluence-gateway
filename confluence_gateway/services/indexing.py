import logging
from pathlib import Path
from typing import Any, Optional

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
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
from confluence_gateway.services.parsers import (
    ContentParser,
    ParserNotAvailableError,
    get_parser,
)

logger = logging.getLogger(__name__)


# Optional dependency imports (markitdown, unstructured) are no longer needed here.
# Dependency checks are handled within the parser classes and factory.

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
        self.html_parser: Optional[ContentParser] = None
        self.attachment_parser: Optional[ContentParser] = None

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

        # Initialize parsers based on config
        try:
            self.html_parser = get_parser(
                parser_name=self.indexing_config.html_parser,
                content_category="html",
            )
            logger.info(
                f"Successfully initialized HTML parser: {self.indexing_config.html_parser}"
            )
        except (ParserNotAvailableError, ValueError) as e:
            logger.warning(
                f"Could not initialize HTML parser '{self.indexing_config.html_parser}': {e}. HTML content parsing will be disabled."
            )
            self.html_parser = None  # Ensure it's None

        try:
            self.attachment_parser = get_parser(
                parser_name=self.indexing_config.attachment_parser,
                content_category="attachment",
            )
            logger.info(
                f"Successfully initialized Attachment parser: {self.indexing_config.attachment_parser}"
            )
        except (ParserNotAvailableError, ValueError) as e:
            logger.warning(
                f"Could not initialize Attachment parser '{self.indexing_config.attachment_parser}': {e}. Attachment parsing will be disabled."
            )
            self.attachment_parser = None  # Ensure it's None

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

        return all_pages

    def _simulate_chunking(self, text: str, chunk_size: int = 200) -> list[str]:
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _process_attachment(
        self, attachment: ConfluenceAttachment, parent_page_id: str
    ) -> Optional[str]:
        """
        Downloads attachment content, checks filters, extracts text using the configured parser.

        Args:
            attachment: The ConfluenceAttachment object.
            parent_page_id: The ID of the page the attachment belongs to.

        Returns:
            The extracted text if successful and filters pass, None otherwise.
        """
        # Need Path for extension checking and io for BytesIO

        attachment_id = attachment.id
        filename = attachment.title
        logger.info(
            f"Processing attachment ID: {attachment_id}, Filename: '{filename}' (Parent Page: {parent_page_id})"
        )

        # 1. Check if attachment indexing is enabled
        if not self.indexing_config.include_attachments:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) because attachment indexing is disabled."
            )
            return None

        # 2. Check if attachment parser is available
        if not self.attachment_parser:
            logger.warning(
                f"Attachment parser not available. Skipping attachment '{filename}' ({attachment_id})."
            )
            return None

        # 3. Check file extension
        file_extension = Path(filename).suffix.lower().lstrip(".")
        allowed_extensions = self.indexing_config.allowed_attachment_extensions
        if allowed_extensions and file_extension not in allowed_extensions:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) due to disallowed extension '{file_extension}'."
            )
            return None

        # 4. Check file size
        file_size_bytes = attachment.file_size
        max_size_bytes = self.indexing_config.max_attachment_size_mb * 1024 * 1024
        if file_size_bytes is None:
            logger.warning(
                f"File size not available for attachment '{filename}' ({attachment_id}). Skipping size check."
            )
        elif file_size_bytes > max_size_bytes:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) because its size ({file_size_bytes / (1024 * 1024):.2f} MB) exceeds the limit ({self.indexing_config.max_attachment_size_mb} MB)."
            )
            return None

        # 5. Download content
        try:
            logger.debug(
                f"Downloading content for attachment '{filename}' ({attachment_id})..."
            )
            attachment_content = self.confluence_client.download_attachment(
                attachment_id
            )
            logger.debug(
                f"Successfully downloaded attachment '{filename}' ({attachment_id})."
            )
        except (ConfluenceAPIError, ConfluenceConnectionError) as e:
            logger.error(
                f"Failed to download attachment '{filename}' ({attachment_id}): {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error downloading attachment '{filename}' ({attachment_id}): {e}",
                exc_info=True,
            )
            return None

        if not attachment_content:
            logger.warning(
                f"Downloaded content for attachment '{filename}' ({attachment_id}) is empty."
            )
            return None

        # 6. Parse content
        try:
            logger.debug(
                f"Parsing content of attachment '{filename}' ({attachment_id})..."
            )
            extracted_text = self.attachment_parser.parse(
                content=attachment_content,
                filename=filename,
                content_type=attachment.media_type,
            )

            if not extracted_text:
                logger.warning(
                    f"Attachment parser yielded no content for '{filename}' ({attachment_id})."
                )
                return None

            logger.info(
                f"Successfully extracted text from attachment '{filename}' ({attachment_id}) (length: {len(extracted_text)})"
            )
            return extracted_text

        except Exception as e:
            logger.error(
                f"Failed to parse attachment '{filename}' ({attachment_id}) using {self.indexing_config.attachment_parser}: {e}",
                exc_info=True,
            )
            return None

    def _process_page(self, page_summary: ConfluencePage) -> Optional[str]:
        """
        Fetches full page details, extracts text, and prepares for indexing.
        This is a partial implementation for Task 1.2 - will be expanded in Task 1.5.

        Returns the extracted text if successful, None otherwise.
        """
        page_id = page_summary.id
        logger.info(f"Processing page ID: {page_id}, Title: '{page_summary.title}'")

        try:
            # Task 1.2.1: Fetch full page content with storage format
            logger.debug(f"Fetching full details for page {page_id}...")
            page_details = self.confluence_client.get_page(
                page_id, expand=["body.storage", "version", "space"]
            )
            logger.debug(f"Successfully fetched details for page {page_id}")

            # Extract storage content (preferred for cleaner HTML)
            html_content = page_details.storage_content
            if not html_content:
                logger.warning(
                    f"No storage content found for page {page_id}. Skipping text extraction."
                )
                return None

            # Task 1.2.3 / 2.2.3: Extract text using the configured parser instance
            extracted_text = None
            if self.html_parser:
                extracted_text = self.html_parser.parse(html_content)
            else:
                logger.warning(
                    f"HTML parser not available for page {page_id}. Skipping text extraction."
                )
                return None  # Cannot proceed without a parser

            if not extracted_text:
                logger.warning(f"HTML parser yielded no content for page {page_id}")
                return None

            logger.info(
                f"Successfully extracted text for page {page_id} (length: {len(extracted_text)})"
            )
            return extracted_text

        except (ConfluenceAPIError, ConfluenceConnectionError) as e:
            logger.error(f"Failed to fetch/process page {page_id}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error processing page {page_id}: {e}", exc_info=True
            )
            return None

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
