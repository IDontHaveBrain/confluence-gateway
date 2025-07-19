"""
ContentFetcher for handling Confluence content retrieval operations.

Extracted from IndexingService to follow Single Responsibility Principle.
Manages fetching of spaces, pages, and content from Confluence API.
"""

import logging

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ConfluenceSpace,
)
from confluence_gateway.core.config import IndexingConfig
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceConnectionError,
)

logger = logging.getLogger(__name__)

PAGE_FETCH_LIMIT = 50


class ContentFetcher:
    """
    Manages Confluence content retrieval operations.

    Handles fetching of spaces, pages, and content from Confluence API.
    Extracted from IndexingService to improve maintainability and testability.

    Responsibilities:
    - Fetch all accessible spaces from Confluence
    - Filter spaces based on include/exclude configuration
    - Fetch pages within specific spaces using CQL queries
    - Handle pagination for large result sets
    """

    def __init__(
        self,
        *,
        confluence_client: ConfluenceClient,
        indexing_config: IndexingConfig,
    ):
        """
        Initialize the ContentFetcher.

        Args:
            confluence_client: Client for Confluence API operations
            indexing_config: Configuration for indexing operations
        """
        self.confluence_client = confluence_client
        self.indexing_config = indexing_config

        logger.debug("ContentFetcher initialized successfully")

    def list_all_accessible_spaces(self) -> list[ConfluenceSpace]:
        """
        List all accessible spaces from Confluence.

        Returns:
            List of all accessible Confluence spaces
        """
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

    def list_target_spaces(self) -> list[ConfluenceSpace]:
        """
        List target spaces based on include/exclude configuration.

        Filters all accessible spaces based on the indexing configuration's
        include_spaces and exclude_spaces settings.

        Returns:
            List of target spaces for indexing
        """
        all_spaces = self.list_all_accessible_spaces()
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

    def list_pages_in_space(self, space_key: str) -> list[ConfluencePage]:
        """
        List all pages in a specific space.

        Uses CQL queries with pagination to fetch all pages in the specified space.

        Args:
            space_key: The key of the space to fetch pages from

        Returns:
            List of pages in the specified space
        """
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
