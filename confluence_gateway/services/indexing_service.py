import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from llama_index.core.node_parser import SentenceSplitter

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
    ConfluenceSpace,
)
from confluence_gateway.adapters.vector_db import (
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
from confluence_gateway.services.common.initialization_logger import (
    InitializationLogger,
)
from confluence_gateway.services.common.validation_utils import ValidationUtils
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.indexing.cleanup_service import CleanupService
from confluence_gateway.services.indexing.content_fetcher import ContentFetcher
from confluence_gateway.services.indexing.embedding_manager import EmbeddingManager
from confluence_gateway.services.indexing.text_processor import TextProcessor
from confluence_gateway.services.parsers import (
    ContentParser,
    get_parser,
)

logger = logging.getLogger(__name__)


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
        content_fetcher: ContentFetcher | None = None,
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
        self.embedding_manager: EmbeddingManager | None = None
        self.cleanup_service: CleanupService | None = None
        self.text_processor: TextProcessor | None = None
        self.content_fetcher: ContentFetcher | None = content_fetcher

        # Initialize vector DB adapter and text splitter with standardized logging
        if self.vector_db_adapter:
            adapter_config = getattr(self.vector_db_adapter, "config", None)
            if isinstance(adapter_config, VectorDBConfig):
                self.vector_db_config = adapter_config
                adapter_type = self.vector_db_config.type
                InitializationLogger.log_configuration_details(
                    "IndexingService",
                    {"Type": adapter_type},
                    logger,
                    prefix_message="IndexingService initialized with provided Vector DB Adapter",
                )
                self.text_splitter = SentenceSplitter(
                    chunk_size=self.vector_db_config.chunk_size,
                    chunk_overlap=self.vector_db_config.chunk_overlap,
                )
                InitializationLogger.log_configuration_details(
                    "SentenceSplitter",
                    {
                        "chunk_size": self.vector_db_config.chunk_size,
                        "chunk_overlap": self.vector_db_config.chunk_overlap,
                    },
                    logger,
                    prefix_message="Initialized SentenceSplitter using adapter's config",
                )
            else:
                logger.warning(
                    "Vector DB Adapter provided to IndexingService, but its configuration could not be accessed. Cannot initialize SentenceSplitter."
                )

        InitializationLogger.log_component_availability(
            "IndexingService",
            "Vector DB Adapter",
            self.vector_db_adapter is not None,
            logger,
            impact_message="Text splitting and vector operations will be limited.",
        )

        InitializationLogger.log_component_availability(
            "IndexingService",
            "Embedding Service",
            self.embedding_service is not None,
            logger,
            impact_message="Embedding features will be disabled for indexing.",
        )

        # Initialize HTML parser with standardized validation
        self.html_parser = ValidationUtils.validate_parser_initialization(
            get_parser,
            self.indexing_config.html_parser,
            logger,
            factory_kwargs={
                "parser_name": self.indexing_config.html_parser,
                "content_category": "html",
            },
            content_type="HTML",
        )

        # Initialize Attachment parser with standardized validation
        self.attachment_parser = ValidationUtils.validate_parser_initialization(
            get_parser,
            self.indexing_config.attachment_parser,
            logger,
            factory_kwargs={
                "parser_name": self.indexing_config.attachment_parser,
                "content_category": "attachment",
            },
            content_type="Attachment",
        )

        # Initialize TextProcessor
        self.text_processor = TextProcessor(
            confluence_client=self.confluence_client,
            indexing_config=self.indexing_config,
            html_parser=self.html_parser,
            attachment_parser=self.attachment_parser,
        )
        InitializationLogger.log_initialization_success("TextProcessor", logger)

        # Initialize EmbeddingManager with dependency validation
        embedding_dependencies = {
            "embedding_service": self.embedding_service,
            "vector_db_adapter": self.vector_db_adapter,
            "vector_db_config": self.vector_db_config,
        }

        if ValidationUtils.validate_conditional_initialization(
            embedding_dependencies, "EmbeddingManager", logger
        ):
            # Dependencies validated above, assert they are not None
            assert self.vector_db_adapter is not None
            assert self.embedding_service is not None
            assert self.vector_db_config is not None

            self.embedding_manager = EmbeddingManager(
                confluence_client=self.confluence_client,
                vector_db_adapter=self.vector_db_adapter,
                embedding_service=self.embedding_service,
                indexing_config=self.indexing_config,
                vector_db_config=self.vector_db_config,
            )
            InitializationLogger.log_initialization_success("EmbeddingManager", logger)
        else:
            self.embedding_manager = None

        # Initialize CleanupService with dependency validation
        cleanup_dependencies = {
            "vector_db_adapter": self.vector_db_adapter,
            "vector_db_config": self.vector_db_config,
        }

        if ValidationUtils.validate_conditional_initialization(
            cleanup_dependencies, "CleanupService", logger
        ):
            # Dependencies validated above, assert they are not None
            assert self.vector_db_adapter is not None
            assert self.vector_db_config is not None

            self.cleanup_service = CleanupService(
                confluence_client=self.confluence_client,
                vector_db_adapter=self.vector_db_adapter,
                indexing_config=self.indexing_config,
                vector_db_config=self.vector_db_config,
            )
            InitializationLogger.log_initialization_success("CleanupService", logger)
        else:
            self.cleanup_service = None

        # Initialize ContentFetcher if not provided
        if not self.content_fetcher:
            self.content_fetcher = ContentFetcher(
                confluence_client=self.confluence_client,
                indexing_config=self.indexing_config,
            )
            InitializationLogger.log_initialization_success("ContentFetcher", logger)
        else:
            InitializationLogger.log_initialization_success(
                "ContentFetcher", logger, "was provided during initialization"
            )

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._last_run_status,
                "last_run_start_time": self._last_run_start_time,
                "last_run_end_time": self._last_run_end_time,
                "last_error_message": self._last_error_message,
            }

    def _process_attachment(
        self,
        attachment: ConfluenceAttachment,
        parent_page_id: str,
    ) -> str | None:
        """Process an attachment and extract its text content."""
        if not self.text_processor:
            logger.warning(
                "TextProcessor not available. Skipping attachment processing."
            )
            return None
        return self.text_processor.process_attachment(attachment, parent_page_id)

    def _process_page(
        self, page_summary: ConfluencePage
    ) -> tuple[ConfluencePage | None, str | None]:
        """Process a page and extract its text content."""
        if not self.text_processor:
            logger.warning("TextProcessor not available. Skipping page processing.")
            return None, None
        return self.text_processor.process_page(page_summary)

    def index_content(
        self,
        content_object: ConfluencePage | ConfluenceAttachment,
        text_content: str,
    ) -> None:
        """
        Index content by delegating to EmbeddingManager.

        Simplified method that delegates to the extracted EmbeddingManager
        for all embedding and vector DB operations.
        """
        if not self.embedding_manager:
            content_id = content_object.id
            document_type = (
                "attachment"
                if isinstance(content_object, ConfluenceAttachment)
                else "page"
            )
            logger.warning(
                f"EmbeddingManager not available. Skipping indexing for {document_type} ID: {content_id}"
            )
            return

        # Delegate to EmbeddingManager
        self.embedding_manager.index_content(content_object, text_content)

    def _delete_vector_db_content(self, content_id: str) -> None:
        """
        Delete content from vector DB by delegating to EmbeddingManager.

        Simplified method that delegates to the extracted EmbeddingManager
        for vector DB deletion operations.
        """
        if not self.embedding_manager:
            logger.warning(
                f"EmbeddingManager not available. Cannot delete content {content_id}"
            )
            return

        # Delegate to EmbeddingManager
        self.embedding_manager.delete_content(content_id)

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
        """
        Clean up deleted content for a space by delegating to CleanupService.

        Simplified method that delegates to the extracted CleanupService
        for all vector DB cleanup operations.
        """
        if not self.cleanup_service:
            logger.warning(
                f"CleanupService not available. Skipping cleanup for space '{space_key}'"
            )
            return

        # Delegate to CleanupService
        cleanup_results = self.cleanup_service.cleanup_deleted_content_for_space(
            space_key, processed_content_ids_in_space
        )

        if cleanup_results["success"]:
            logger.info(
                f"Cleanup completed for space '{space_key}': "
                f"Found {cleanup_results['orphaned_count']} orphaned content IDs, "
                f"deleted {cleanup_results['deleted_count']}, "
                f"failed {cleanup_results['failed_count']}"
            )
        else:
            logger.error(
                f"Cleanup failed for space '{space_key}': {cleanup_results.get('error', 'Unknown error')}"
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

        # Flag to track successful completion
        success_flag = False

        try:
            await asyncio.to_thread(self._run_indexing_sync, space_keys, index_all)
            success_flag = True
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

                # Set success status only if operation completed successfully
                if success_flag:
                    self._last_run_status = "success"
                    self._last_error_message = None
                # Safety check: if status is still "running", something unexpected happened
                elif self._last_run_status == "running":
                    self._last_run_status = "failure"
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
            if not self.content_fetcher:
                logger.error(
                    "ContentFetcher is not available. Cannot proceed with indexing."
                )
                return
            target_spaces = self.content_fetcher.list_all_accessible_spaces()
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
            if not self.content_fetcher:
                logger.error(
                    "ContentFetcher is not available. Cannot proceed with indexing."
                )
                return
            target_spaces = self.content_fetcher.list_target_spaces()

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

            if not self.content_fetcher:
                logger.error(
                    "ContentFetcher is not available. Cannot proceed with indexing."
                )
                return
            pages_in_space = self.content_fetcher.list_pages_in_space(space_key)
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
