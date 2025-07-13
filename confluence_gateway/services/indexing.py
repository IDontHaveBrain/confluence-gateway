import asyncio
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from llama_index.core.node_parser import SentenceSplitter

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


PAGE_FETCH_LIMIT = 50


class IndexingService:
    _instance: Optional["IndexingService"] = None
    _lock = threading.Lock()

    _is_running: bool = False
    _last_run_start_time: datetime | None = None
    _last_run_end_time: datetime | None = None
    _last_run_status: Literal["idle", "running", "success", "failure"] = "idle"
    _last_error_message: str | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "IndexingService":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        confluence_client: ConfluenceClient,
        indexing_config: IndexingConfig,
        search_config: SearchConfig,
        embedding_service: EmbeddingService | None = None,
        vector_db_adapter: VectorDBAdapter | None = None,
    ):
        self.confluence_client = confluence_client
        self.indexing_config = indexing_config
        self.search_config = search_config
        self.embedding_service = embedding_service
        self.vector_db_adapter: VectorDBAdapter | None = vector_db_adapter
        self.vector_db_config: VectorDBConfig | None = None
        self.text_splitter: SentenceSplitter | None = None
        self.html_parser: ContentParser | None = None
        self.attachment_parser: ContentParser | None = None

        if self.vector_db_adapter:
            adapter_config = getattr(self.vector_db_adapter, "config", None)
            if isinstance(adapter_config, VectorDBConfig):
                self.vector_db_config = adapter_config
                adapter_type = self.vector_db_config.type
                logger.info(
                    f"IndexingService initialized with provided Vector DB Adapter: Type='{adapter_type}'"
                )
                self.text_splitter = SentenceSplitter(
                    chunk_size=self.vector_db_config.chunk_size,
                    chunk_overlap=self.vector_db_config.chunk_overlap,
                )
                logger.info(
                    f"Initialized SentenceSplitter using adapter's config: chunk_size={self.vector_db_config.chunk_size}, chunk_overlap={self.vector_db_config.chunk_overlap}"
                )
            else:
                logger.warning(
                    "Vector DB Adapter provided to IndexingService, but its configuration could not be accessed. Cannot initialize SentenceSplitter."
                )
        else:
            logger.warning(
                "IndexingService initialized WITHOUT Vector DB Adapter (adapter not provided)."
            )

        if self.embedding_service:
            logger.info("IndexingService initialized with Embedding Service.")
        else:
            logger.warning(
                "IndexingService initialized WITHOUT Embedding Service. Embedding features will be disabled for indexing."
            )

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
            self.html_parser = None

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
            self.attachment_parser = None

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._last_run_status,
                "last_run_start_time": self._last_run_start_time,
                "last_run_end_time": self._last_run_end_time,
                "last_error_message": self._last_error_message,
            }

    def _list_all_accessible_spaces(self) -> list[ConfluenceSpace]:
        try:
            logger.info("Fetching all accessible spaces from Confluence via client...")
            all_spaces: list[ConfluenceSpace] = self.confluence_client.list_all_spaces()
            logger.info(f"Client returned {len(all_spaces)} accessible spaces.")
            return all_spaces
        except (ConfluenceAPIError, ConfluenceConnectionError) as e:
            logger.error(f"Failed to fetch spaces from Confluence: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while fetching spaces: {e}",
                exc_info=True,
            )
            return []

    def _list_target_spaces(self) -> list[ConfluenceSpace]:
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
        all_pages: list[ConfluencePage] = []
        start = 0
        limit = PAGE_FETCH_LIMIT
        # Note: We rely on the client/library to handle potential special characters in space_key if necessary.
        # If space keys can contain quotes or other CQL special chars, client-side escaping might be needed.
        cql = f'space = "{space_key}" AND type = page'
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
                    get_all_results=False,
                    expand=["version"],
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
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching pages for space '{space_key}' at start={start}: {e}",
                    exc_info=True,
                )
                break

        return all_pages

    def _chunk_text(self, text: str) -> list[str]:
        if not self.text_splitter:
            logger.warning(
                "Text splitter not initialized. Returning text as a single chunk."
            )
            return [text] if text else []
        if not text:
            return []
        return list(self.text_splitter.split_text(text))

    def _create_chunk_metadata(
        self,
        *,
        content_object: ConfluencePage | ConfluenceAttachment,
        document_type: str,
        chunk_sequence_number: int,
    ) -> dict[str, Any]:
        base_fields = self.confluence_client.extract_content_fields(content_object)
        metadata = {
            "document_type": document_type,
            "original_content_id": base_fields.get("id"),
            "title": base_fields.get("title"),
            "space_key": base_fields.get("space_key"),
            "url": base_fields.get("url"),
            "last_modified": (
                dt.isoformat()
                if (
                    dt := base_fields.get("updated_at") or base_fields.get("created_at")
                )
                else None
            ),
            "chunk_sequence_number": chunk_sequence_number,
        }

        if document_type == "attachment":
            metadata["parent_page_id"] = base_fields.get("parent_id")
            metadata["attachment_filename"] = base_fields.get("file_name")
            if not metadata.get("parent_page_id") and hasattr(
                content_object, "container"
            ):
                container = getattr(content_object, "container", None)
                if container:
                    metadata["parent_page_id"] = (
                        container.get("id", "")
                        if isinstance(container, dict)
                        else getattr(container, "id", "")
                    )

        return {k: v for k, v in metadata.items() if v is not None}

    def _process_attachment(
        self,
        attachment: ConfluenceAttachment,
        parent_page_id: str,
    ) -> str | None:
        attachment_id = attachment.id
        filename = attachment.title
        logger.info(
            f"Processing attachment ID: {attachment_id}, Filename: '{filename}' (Parent Page: {parent_page_id})"
        )

        if not self.indexing_config.include_attachments:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) because attachment indexing is disabled."
            )
            return None

        if not self.attachment_parser:
            logger.warning(
                f"Attachment parser not available. Skipping attachment '{filename}' ({attachment_id})."
            )
            return None

        file_extension = Path(filename).suffix.lower().lstrip(".")
        allowed_extensions = self.indexing_config.allowed_attachment_extensions
        if allowed_extensions and file_extension not in allowed_extensions:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) due to disallowed extension '{file_extension}'."
            )
            return None

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

    def _process_page(
        self, page_summary: ConfluencePage
    ) -> tuple[ConfluencePage | None, str | None]:
        page_id = page_summary.id
        logger.info(f"Processing page ID: {page_id}, Title: '{page_summary.title}'")
        page_details: ConfluencePage | None = None

        try:
            logger.debug(f"Fetching full details for page {page_id}...")
            page_details = self.confluence_client.get_page(
                page_id, expand=["body.storage", "version", "space"]
            )
            logger.debug(f"Successfully fetched details for page {page_id}")

            html_content = page_details.storage_content
            if not html_content:
                logger.warning(
                    f"No storage content found for page {page_id}. Skipping text extraction."
                )
                return page_details, None

            extracted_text = None
            if self.html_parser:
                logger.debug(f"Parsing HTML content for page {page_id}...")
                extracted_text = self.html_parser.parse(html_content)
            else:
                logger.warning(
                    f"HTML parser not available for page {page_id}. Skipping text extraction."
                )
                return page_details, None

            if not extracted_text:
                logger.warning(f"HTML parser yielded no content for page {page_id}")
                return page_details, None

            logger.info(
                f"Successfully extracted text for page {page_id} (length: {len(extracted_text)})"
            )
            return page_details, extracted_text

        except (ConfluenceAPIError, ConfluenceConnectionError) as e:
            logger.error(f"Failed to fetch/process page {page_id}: {e}", exc_info=True)
            return page_details, None
        except Exception as e:
            logger.error(
                f"Unexpected error processing page {page_id}: {e}", exc_info=True
            )
            return page_details, None

    def index_content(
        self,
        content_object: ConfluencePage | ConfluenceAttachment,
        text_content: str,
    ) -> None:
        content_id = content_object.id
        document_type = (
            "attachment" if isinstance(content_object, ConfluenceAttachment) else "page"
        )

        if not self.vector_db_adapter:
            logger.warning(
                f"Vector DB Adapter not available. Skipping indexing for {document_type} ID: {content_id}"
            )
            return
        if not self.embedding_service:
            logger.warning(
                f"Embedding Service not available. Skipping indexing for {document_type} ID: {content_id}"
            )
            return
        if not self.text_splitter:
            logger.warning(
                f"Text Splitter not available. Skipping indexing for {document_type} ID: {content_id}"
            )
            return

        if not text_content:
            logger.warning(
                f"Received empty text content for {document_type} ID: {content_id}. Skipping indexing."
            )
            return

        logger.info(
            f"Starting indexing process for {document_type} ID: {content_id}, Title: '{content_object.title}'"
        )

        chunks = self._chunk_text(text_content)
        if not chunks:
            logger.warning(
                f"No chunks generated from text content for {document_type} ID: {content_id}"
            )
            return
        logger.debug(
            f"Generated {len(chunks)} chunks for {document_type} ID: {content_id}"
        )

        try:
            logger.info(
                f"Generating embeddings for {len(chunks)} chunks for {document_type} ID: {content_id}..."
            )
            embeddings = self.embedding_service.embed_texts(chunks)
            logger.info(
                f"Successfully generated {len(embeddings)} embeddings for {document_type} ID: {content_id}."
            )

            if len(embeddings) != len(chunks):
                logger.error(
                    f"Mismatch between number of chunks ({len(chunks)}) and generated embeddings ({len(embeddings)}) for {document_type} ID: {content_id}. Skipping upsert."
                )
                return

        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embeddings for {document_type} ID {content_id}: {e}",
                exc_info=True,
            )
            return
        except Exception as e:
            logger.error(
                f"Unexpected error during embedding generation for {document_type} ID {content_id}: {e}",
                exc_info=True,
            )
            return

        documents_to_upsert: list[Document] = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None or not embedding:
                logger.warning(
                    f"Skipping chunk {i} for {document_type} {content_id} due to missing/empty embedding."
                )
                continue

            chunk_uuid = str(uuid.uuid4())
            chunk_metadata = self._create_chunk_metadata(
                content_object=content_object,
                document_type=document_type,
                chunk_sequence_number=i,
            )
            if "original_content_id" not in chunk_metadata:
                chunk_metadata["original_content_id"] = content_id

            doc = Document(
                id=chunk_uuid,
                text=chunk_text,
                embedding=embedding,
                metadata=chunk_metadata,
            )
            documents_to_upsert.append(doc)

        if not documents_to_upsert:
            logger.warning(
                f"No valid documents prepared for upsert for {document_type} ID: {content_id}"
            )
            return

        try:
            assert self.vector_db_config is not None
            logger.info(
                f"Upserting {len(documents_to_upsert)} documents for {document_type} ID: {content_id} using {self.vector_db_config.type} adapter."
            )
            self.vector_db_adapter.upsert(documents=documents_to_upsert)
            logger.info(
                f"Successfully upserted documents for {document_type} ID: {content_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to upsert documents for {document_type} ID {content_id}: {e}",
                exc_info=True,
            )

    def _delete_vector_db_content(self, content_id: str) -> None:
        if not self.vector_db_adapter:
            logger.warning(
                f"Cannot delete content {content_id}: Vector DB Adapter not available."
            )
            return

        logger.info(f"Deleting existing vector DB entries for content ID: {content_id}")
        try:
            self.vector_db_adapter.delete_by_metadata(
                filters={"original_content_id": content_id}
            )
            logger.info(
                f"Successfully submitted deletion request for content ID: {content_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete vector DB entries for content ID {content_id}: {e}",
                exc_info=True,
            )

    def _should_index_content(
        self, content_object: ConfluencePage | ConfluenceAttachment
    ) -> tuple[bool, bool]:
        content_id = content_object.id
        doc_type = (
            "attachment" if isinstance(content_object, ConfluenceAttachment) else "page"
        )

        if not self.vector_db_adapter:
            logger.debug(
                f"Skipping timestamp check for {doc_type} {content_id}: Vector DB Adapter not available. Assuming indexing is needed."
            )
            return True, False

        current_last_modified = content_object.updated_at or content_object.created_at
        if not current_last_modified:
            logger.warning(
                f"Could not determine current last_modified time for {doc_type} {content_id}. Assuming re-indexing is needed."
            )
            return True, True

        if current_last_modified.tzinfo is None:
            current_last_modified = current_last_modified.replace(tzinfo=timezone.utc)

        try:
            logger.debug(
                f"Checking stored timestamp for {doc_type} {content_id} in vector DB..."
            )
            stored_data = self.vector_db_adapter.search_by_metadata(
                filters={"original_content_id": content_id},
                select=["last_modified"],
                limit=1,
            )

            if not stored_data:
                logger.info(
                    f"{doc_type.capitalize()} {content_id} not found in vector DB. Needs indexing."
                )
                return True, False

            stored_timestamp_str = stored_data[0].get("last_modified")
            if not stored_timestamp_str:
                logger.warning(
                    f"Found entry for {doc_type} {content_id} in vector DB, but 'last_modified' metadata is missing. Assuming re-indexing is needed."
                )
                return True, True

            try:
                stored_last_modified = datetime.fromisoformat(stored_timestamp_str)
                if stored_last_modified.tzinfo is None:
                    stored_last_modified = stored_last_modified.replace(
                        tzinfo=timezone.utc
                    )

            except ValueError:
                logger.warning(
                    f"Could not parse stored last_modified timestamp '{stored_timestamp_str}' for {doc_type} {content_id}. Assuming re-indexing is needed."
                )
                return True, True

            tolerance_seconds = 1
            if current_last_modified > stored_last_modified + timedelta(
                seconds=tolerance_seconds
            ):
                logger.info(
                    f"{doc_type.capitalize()} {content_id} has been modified ({current_last_modified} > {stored_last_modified}). Needs re-indexing."
                )
                return True, True
            else:
                logger.info(
                    f"{doc_type.capitalize()} {content_id} is up-to-date ({current_last_modified} <= {stored_last_modified}). Skipping indexing."
                )
                return False, False

        except Exception as e:
            logger.error(
                f"Error checking stored timestamp for {doc_type} {content_id}: {e}. Assuming re-indexing is needed.",
                exc_info=True,
            )
            return True, True

    def _cleanup_deleted_content_for_space(
        self, space_key: str, processed_content_ids_in_space: set[str]
    ) -> None:
        if not self.vector_db_adapter:
            logger.warning(
                f"Skipping deleted content cleanup for space '{space_key}': Vector DB Adapter not available."
            )
            return

        logger.info(
            f"Starting cleanup check for deleted content in space '{space_key}'..."
        )

        try:
            logger.debug(
                f"Fetching stored content IDs for space '{space_key}' from vector DB..."
            )
            stored_metadata = self.vector_db_adapter.search_by_metadata(
                filters={"space_key": space_key},
                select=["original_content_id"],
            )

            if not stored_metadata:
                logger.info(
                    f"No content found in vector DB for space '{space_key}'. No cleanup needed."
                )
                return

            db_content_ids = {
                item["original_content_id"]
                for item in stored_metadata
                if "original_content_id" in item
            }
            logger.debug(
                f"Found {len(db_content_ids)} unique content IDs in vector DB for space '{space_key}'."
            )
            logger.debug(
                f"Processed {len(processed_content_ids_in_space)} content IDs from Confluence in this run for space '{space_key}'."
            )

            orphaned_ids = db_content_ids - processed_content_ids_in_space

            if orphaned_ids:
                logger.info(
                    f"Found {len(orphaned_ids)} orphaned content IDs in space '{space_key}' to delete: {orphaned_ids}"
                )
                deleted_count = 0
                failed_count = 0
                for content_id in orphaned_ids:
                    try:
                        self._delete_vector_db_content(content_id)
                        deleted_count += 1
                    except Exception:
                        failed_count += 1
                logger.info(
                    f"Cleanup for space '{space_key}': Submitted deletion for {deleted_count} orphaned IDs, {failed_count} failures."
                )
            else:
                logger.info(
                    f"No orphaned content found for space '{space_key}'. Cleanup complete."
                )

        except Exception as e:
            logger.error(
                f"Error during deleted content cleanup check for space '{space_key}': {e}. Cleanup for this space aborted.",
                exc_info=True,
            )

    async def run_indexing(
        self, space_keys: list[str] | None = None, index_all: bool = False
    ) -> None:
        with self._lock:
            if self._is_running:
                logger.warning("Indexing is already running. Skipping new trigger.")
                return
            self._is_running = True
            self._last_run_start_time = datetime.now(timezone.utc)
            self._last_run_end_time = None
            self._last_run_status = "running"
            self._last_error_message = None

        if index_all:
            target_description = "All accessible spaces"
        elif space_keys:
            target_description = f"{space_keys}"
        else:
            target_description = "Configured spaces"

        logger.info(
            f"Scheduling indexing run in background thread... Target: {target_description}"
        )
        try:
            await asyncio.to_thread(self._run_indexing_sync, space_keys, index_all)
            with self._lock:
                self._last_run_status = "success"
                self._last_error_message = None
            logger.info("Background indexing run completed successfully.")
        except Exception as e:
            error_msg = f"Background indexing run failed: {e}"
            logger.error(error_msg, exc_info=True)
            with self._lock:
                self._last_run_status = "failure"
                self._last_error_message = error_msg
        finally:
            with self._lock:
                self._is_running = False
                self._last_run_end_time = datetime.now(timezone.utc)
                if self._last_run_status == "running":
                    self._last_run_status = "failure"  # type: ignore[unreachable]
                    self._last_error_message = "Indexing finished unexpectedly without success or failure status."

    def _run_indexing_sync(
        self, space_keys: list[str] | None = None, index_all: bool = False
    ) -> None:
        logger.info("Background indexing thread started.")

        if not self.vector_db_adapter or not self.embedding_service:
            logger.error(
                "Indexing cannot proceed: Vector DB Adapter or Embedding Service is not available."
            )
            return

        target_spaces: list[ConfluenceSpace] = []
        if index_all:
            logger.info("Fetching all accessible spaces for indexing...")
            target_spaces = self._list_all_accessible_spaces()
        elif space_keys:
            logger.info(f"Targeting specific spaces provided: {space_keys}")
            for key in space_keys:
                try:
                    space = self.confluence_client.get_space(key)
                    target_spaces.append(space)
                except (ConfluenceAPIError, ConfluenceConnectionError) as e:
                    logger.error(
                        f"Could not fetch details for specified space key '{key}': {e}. Skipping."
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error fetching details for space key '{key}': {e}. Skipping."
                    )
        else:
            logger.info("Determining target spaces based on configuration...")
            target_spaces = self._list_target_spaces()

        if not target_spaces:
            logger.warning("No target spaces found to index.")
            logger.info("Indexing run finished (no spaces).")
            return

        logger.info(
            f"Starting indexing for {len(target_spaces)} spaces: {[s.key for s in target_spaces]}"
        )

        for space in target_spaces:
            space_key = space.key
            logger.info(f"--- Processing Space: {space_key} ({space.title}) ---")
            processed_content_ids_in_space: set[str] = set()

            pages_in_space = self._list_pages_in_space(space_key)
            logger.info(f"Found {len(pages_in_space)} pages in space {space_key}.")

            for page_summary in pages_in_space:
                page_id = page_summary.id
                logger.info(
                    f"--- Checking Page ID: {page_id} ('{page_summary.title}') ---"
                )
                processed_content_ids_in_space.add(page_id)

                page_details_for_check: ConfluencePage | None = None
                try:
                    page_details_for_check = self.confluence_client.get_page(
                        page_id,
                        expand=[
                            "version",
                            "space",
                        ],
                    )
                except (ConfluenceAPIError, ConfluenceConnectionError) as e:
                    logger.error(
                        f"Failed to fetch basic details for page {page_id} check: {e}. Skipping page."
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Unexpected error fetching basic details for page {page_id} check: {e}. Skipping page."
                    )
                    continue

                if not page_details_for_check:
                    logger.warning(
                        f"Could not retrieve details for page {page_id} check. Skipping page."
                    )
                    continue

                should_index_page, page_needs_delete = self._should_index_content(
                    page_details_for_check
                )

                page_details_full: ConfluencePage | None = None

                if should_index_page:
                    logger.info(f"Page {page_id}: Processing required.")
                    if page_needs_delete:
                        self._delete_vector_db_content(page_id)

                    try:
                        page_details_full, extracted_page_text = self._process_page(
                            page_summary
                        )

                        if page_details_full and extracted_page_text:
                            logger.debug(
                                f"Page {page_id}: Text extracted, proceeding to index."
                            )
                            self.index_content(page_details_full, extracted_page_text)
                        elif page_details_full:
                            logger.debug(
                                f"Page {page_id}: No text extracted or parser failed, skipping page content indexing."
                            )
                        else:
                            logger.warning(
                                f"Page {page_id}: Failed to fetch full page details after check. Skipping."
                            )
                            continue

                    except Exception as page_proc_err:
                        logger.error(
                            f"Unexpected error during full processing of page {page_id}: {page_proc_err}",
                            exc_info=True,
                        )
                        continue
                else:
                    logger.info(f"Page {page_id}: Skipping processing (up-to-date).")
                    page_details_full = page_details_for_check

                if page_details_full and self.indexing_config.include_attachments:
                    logger.debug(f"Listing attachments for page {page_id}...")
                    try:
                        attachments = self.confluence_client.list_attachments(page_id)
                        logger.info(
                            f"Found {len(attachments)} attachments for page {page_id}."
                        )

                        for attachment in attachments:
                            attachment_id = attachment.id
                            logger.info(
                                f"--- Checking Attachment ID: {attachment_id} ('{attachment.title}') ---"
                            )
                            processed_content_ids_in_space.add(attachment_id)

                            should_index_attach, attach_needs_delete = (
                                self._should_index_content(attachment)
                            )

                            if should_index_attach:
                                logger.info(
                                    f"Attachment {attachment.id}: Processing required."
                                )
                                if attach_needs_delete:
                                    self._delete_vector_db_content(attachment.id)

                                try:
                                    extracted_attachment_text = (
                                        self._process_attachment(attachment, page_id)
                                    )
                                    if extracted_attachment_text:
                                        logger.debug(
                                            f"Attachment {attachment.id}: Text extracted, proceeding to index."
                                        )
                                        self.index_content(
                                            attachment, extracted_attachment_text
                                        )
                                    else:
                                        logger.debug(
                                            f"Attachment {attachment.id}: No text extracted or skipped by filters/parser."
                                        )
                                except Exception as attach_proc_err:
                                    logger.error(
                                        f"Unexpected error processing attachment {attachment.id} for page {page_id}: {attach_proc_err}",
                                        exc_info=True,
                                    )
                            else:
                                logger.info(
                                    f"Attachment {attachment.id}: Skipping processing (up-to-date)."
                                )

                    except (
                        ConfluenceAPIError,
                        ConfluenceConnectionError,
                    ) as attach_list_err:
                        logger.error(
                            f"Failed to list attachments for page {page_id}: {attach_list_err}",
                            exc_info=True,
                        )
                        # Decide if we should continue processing the space or stop
                        continue  # Continue to the next page in the space

            logger.info(
                f"--- Finished Processing Pages/Attachments for Space: {space_key} ---"
            )

            self._cleanup_deleted_content_for_space(
                space_key, processed_content_ids_in_space
            )

            logger.info(f"--- Completed All Tasks for Space: {space_key} ---")

        logger.info("Background indexing thread finished.")
