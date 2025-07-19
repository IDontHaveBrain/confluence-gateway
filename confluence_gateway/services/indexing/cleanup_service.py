"""
CleanupService for handling vector database cleanup operations.

Extracted from IndexingService to follow Single Responsibility Principle.
Manages cleanup of deleted content from vector database.
"""

import logging
from typing import Any

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.vector_db import VectorDBAdapter
from confluence_gateway.core.config import IndexingConfig, VectorDBConfig

logger = logging.getLogger(__name__)


class CleanupService:
    """
    Manages cleanup operations for vector database content.

    Handles identification and removal of orphaned content from the vector database
    when content is deleted from Confluence. Extracted from IndexingService
    to improve maintainability and testability.

    Responsibilities:
    - Compare vector DB content with current Confluence content
    - Identify orphaned content that no longer exists in Confluence
    - Remove orphaned content from vector database
    - Report cleanup statistics and errors
    """

    def __init__(
        self,
        *,
        confluence_client: ConfluenceClient,
        vector_db_adapter: VectorDBAdapter,
        indexing_config: IndexingConfig,
        vector_db_config: VectorDBConfig,
    ):
        """
        Initialize the CleanupService.

        Args:
            confluence_client: Client for Confluence API operations
            vector_db_adapter: Adapter for vector database operations
            indexing_config: Configuration for indexing operations
            vector_db_config: Configuration for vector database
        """
        self.confluence_client = confluence_client
        self.vector_db_adapter = vector_db_adapter
        self.indexing_config = indexing_config
        self.vector_db_config = vector_db_config

        logger.debug("CleanupService initialized")

    def cleanup_deleted_content_for_space(
        self, space_key: str, processed_content_ids_in_space: set[str]
    ) -> dict[str, Any]:
        """
        Clean up deleted content for a specific space.

        Compares the content IDs that were processed during the current indexing run
        with the content IDs stored in the vector database for the given space.
        Any content IDs that exist in the vector DB but were not processed
        (indicating they were deleted from Confluence) are removed from the vector DB.

        Args:
            space_key: Key of the Confluence space to clean up
            processed_content_ids_in_space: Set of content IDs that were processed
                                          during the current indexing run

        Returns:
            Dictionary with cleanup statistics:
            - 'orphaned_count': Number of orphaned content IDs found
            - 'deleted_count': Number of successfully deleted content IDs
            - 'failed_count': Number of content IDs that failed to delete
            - 'success': Whether the cleanup operation completed successfully
        """
        if not self.vector_db_adapter:
            logger.warning(
                f"Skipping deleted content cleanup for space '{space_key}': Vector DB Adapter not available."
            )
            return {
                "orphaned_count": 0,
                "deleted_count": 0,
                "failed_count": 0,
                "success": False,
                "error": "Vector DB Adapter not available",
            }

        logger.info(
            f"Starting cleanup check for deleted content in space '{space_key}'..."
        )

        try:
            # Get stored content IDs from vector DB
            stored_content_ids = self._get_stored_content_ids(space_key)

            if not stored_content_ids:
                logger.info(
                    f"No content found in vector DB for space '{space_key}'. No cleanup needed."
                )
                return {
                    "orphaned_count": 0,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "success": True,
                }

            # Identify orphaned content
            orphaned_ids = self._identify_orphaned_content(
                stored_content_ids, processed_content_ids_in_space, space_key
            )

            if not orphaned_ids:
                logger.info(
                    f"No orphaned content found for space '{space_key}'. Cleanup complete."
                )
                return {
                    "orphaned_count": 0,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "success": True,
                }

            # Delete orphaned content
            deletion_results = self._delete_orphaned_content(orphaned_ids, space_key)

            logger.info(
                f"Cleanup for space '{space_key}': Found {len(orphaned_ids)} orphaned IDs, "
                f"deleted {deletion_results['deleted_count']}, "
                f"failed {deletion_results['failed_count']}"
            )

            return {
                "orphaned_count": len(orphaned_ids),
                "deleted_count": deletion_results["deleted_count"],
                "failed_count": deletion_results["failed_count"],
                "success": True,
            }

        except Exception as e:
            logger.error(
                f"Error during deleted content cleanup check for space '{space_key}': {e}. "
                f"Cleanup for this space aborted.",
                exc_info=True,
            )
            return {
                "orphaned_count": 0,
                "deleted_count": 0,
                "failed_count": 0,
                "success": False,
                "error": str(e),
            }

    def _get_stored_content_ids(self, space_key: str) -> set[str]:
        """
        Get content IDs stored in vector DB for a specific space.

        Args:
            space_key: Key of the Confluence space

        Returns:
            Set of content IDs found in the vector database
        """
        logger.debug(
            f"Fetching stored content IDs for space '{space_key}' from vector DB..."
        )

        stored_metadata = self.vector_db_adapter.search_by_metadata(
            filters={"space_key": space_key},
            select=["original_content_id"],
        )

        if not stored_metadata:
            return set()

        content_ids = {
            item["original_content_id"]
            for item in stored_metadata
            if "original_content_id" in item
        }

        logger.debug(
            f"Found {len(content_ids)} unique content IDs in vector DB for space '{space_key}'."
        )

        return content_ids

    def _identify_orphaned_content(
        self,
        stored_content_ids: set[str],
        processed_content_ids_in_space: set[str],
        space_key: str,
    ) -> set[str]:
        """
        Identify orphaned content by comparing stored and processed content IDs.

        Args:
            stored_content_ids: Content IDs found in vector DB
            processed_content_ids_in_space: Content IDs processed from Confluence
            space_key: Key of the Confluence space (for logging)

        Returns:
            Set of orphaned content IDs that should be deleted
        """
        logger.debug(
            f"Processed {len(processed_content_ids_in_space)} content IDs from Confluence "
            f"in this run for space '{space_key}'."
        )

        orphaned_ids = stored_content_ids - processed_content_ids_in_space

        if orphaned_ids:
            logger.info(
                f"Found {len(orphaned_ids)} orphaned content IDs in space '{space_key}' "
                f"to delete: {orphaned_ids}"
            )

        return orphaned_ids

    def _delete_orphaned_content(
        self, orphaned_ids: set[str], space_key: str
    ) -> dict[str, int]:
        """
        Delete orphaned content from vector database.

        Args:
            orphaned_ids: Set of content IDs to delete
            space_key: Key of the Confluence space (for logging)

        Returns:
            Dictionary with deletion counts:
            - 'deleted_count': Number of successfully deleted content IDs
            - 'failed_count': Number of content IDs that failed to delete
        """
        deleted_count = 0
        failed_count = 0

        for content_id in orphaned_ids:
            try:
                self._delete_vector_db_content(content_id)
                deleted_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to delete orphaned content {content_id} from space '{space_key}': {e}",
                    exc_info=True,
                )
                failed_count += 1

        return {"deleted_count": deleted_count, "failed_count": failed_count}

    def _delete_vector_db_content(self, content_id: str) -> None:
        """
        Delete content from vector DB by content ID.

        Args:
            content_id: ID of the content to delete

        Raises:
            Exception: If deletion fails
        """
        try:
            self.vector_db_adapter.delete_by_metadata(
                {"original_content_id": content_id}
            )
            logger.debug(f"Successfully deleted content {content_id} from vector DB")
        except Exception as e:
            logger.error(f"Failed to delete content {content_id} from vector DB: {e}")
            raise
