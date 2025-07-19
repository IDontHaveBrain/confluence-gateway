"""
TextProcessor for handling content parsing and processing operations.

Extracted from IndexingService to follow Single Responsibility Principle.
Manages content parsing for pages and attachments, including filtering
and validation operations.
"""

import logging
from pathlib import Path
from typing import cast

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
)
from confluence_gateway.core.config import IndexingConfig
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceConnectionError,
)
from confluence_gateway.services.parsers import ContentParser

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles content parsing and processing operations for indexing.

    Manages content parsing for both pages and attachments, including
    filtering by file size and extension, and validation operations.
    Extracted from IndexingService to improve maintainability and testability.

    Responsibilities:
    - Process page content with HTML parsing
    - Process attachment content with file downloading and parsing
    - Apply file size and extension filtering
    - Handle content validation and error cases
    - Provide structured processing results
    """

    def __init__(
        self,
        *,
        confluence_client: ConfluenceClient,
        indexing_config: IndexingConfig,
        html_parser: ContentParser | None = None,
        attachment_parser: ContentParser | None = None,
    ):
        """
        Initialize the TextProcessor.

        Args:
            confluence_client: Client for Confluence API operations
            indexing_config: Configuration for indexing operations
            html_parser: Parser for HTML content (optional)
            attachment_parser: Parser for attachment content (optional)
        """
        self.confluence_client = confluence_client
        self.indexing_config = indexing_config
        self.html_parser = html_parser
        self.attachment_parser = attachment_parser

        logger.debug(
            f"TextProcessor initialized with html_parser={html_parser is not None}, attachment_parser={attachment_parser is not None}"
        )

    def process_attachment(
        self,
        attachment: ConfluenceAttachment,
        parent_page_id: str,
    ) -> str | None:
        """
        Process an attachment and extract its text content.

        Args:
            attachment: The attachment to process
            parent_page_id: ID of the parent page

        Returns:
            Extracted text content or None if processing failed/skipped
        """
        attachment_id = attachment.id
        filename = attachment.title
        logger.info(
            f"Processing attachment ID: {attachment_id}, Filename: '{filename}' (Parent Page: {parent_page_id})"
        )

        # Check if attachment processing is enabled
        if not self.indexing_config.include_attachments:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) because attachment indexing is disabled."
            )
            return None

        # Check if attachment parser is available
        if not self.attachment_parser:
            logger.warning(
                f"Attachment parser not available. Skipping attachment '{filename}' ({attachment_id})."
            )
            return None

        # Validate file extension
        if not self._is_allowed_file_extension(filename):
            file_extension = Path(filename).suffix.lower().lstrip(".")
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment_id}) due to disallowed extension '{file_extension}'."
            )
            return None

        # Validate file size
        if not self._is_allowed_file_size(attachment, filename):
            return None

        # Download attachment content
        attachment_content = self._download_attachment_content(attachment, filename)
        if not attachment_content:
            return None

        # Parse attachment content
        return self._parse_attachment_content(attachment, attachment_content, filename)

    def process_page(
        self, page_summary: ConfluencePage
    ) -> tuple[ConfluencePage | None, str | None]:
        """
        Process a page and extract its text content.

        Args:
            page_summary: The page summary to process

        Returns:
            Tuple of (full page details, extracted text content)
            Either can be None if processing failed
        """
        page_id = page_summary.id
        logger.info(f"Processing page ID: {page_id}, Title: '{page_summary.title}'")
        page_details: ConfluencePage | None = None

        try:
            # Fetch full page details
            logger.debug(f"Fetching full details for page {page_id}...")
            page_details = self.confluence_client.get_page(
                page_id, expand=["body.storage", "version", "space"]
            )
            logger.debug(f"Successfully fetched details for page {page_id}")

            # Extract HTML content
            html_content = page_details.storage_content
            if not html_content:
                logger.warning(
                    f"No storage content found for page {page_id}. Skipping text extraction."
                )
                return page_details, None

            # Parse HTML content
            extracted_text = self._parse_html_content(html_content, page_id)
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

    def _is_allowed_file_extension(self, filename: str) -> bool:
        """Check if file extension is allowed based on configuration."""
        file_extension = Path(filename).suffix.lower().lstrip(".")
        allowed_extensions = self.indexing_config.allowed_attachment_extensions
        return not allowed_extensions or file_extension in allowed_extensions

    def _is_allowed_file_size(
        self, attachment: ConfluenceAttachment, filename: str
    ) -> bool:
        """Check if file size is within allowed limits."""
        file_size_bytes = attachment.file_size
        max_size_bytes = self.indexing_config.max_attachment_size_mb * 1024 * 1024

        if file_size_bytes is None:
            logger.warning(
                f"File size not available for attachment '{filename}' ({attachment.id}). Skipping size check."
            )
            return True

        if file_size_bytes > max_size_bytes:
            logger.debug(
                f"Skipping attachment '{filename}' ({attachment.id}) because its size ({file_size_bytes / (1024 * 1024):.2f} MB) exceeds the limit ({self.indexing_config.max_attachment_size_mb} MB)."
            )
            return False

        return True

    def _download_attachment_content(
        self, attachment: ConfluenceAttachment, filename: str
    ) -> bytes | None:
        """Download attachment content from Confluence."""
        attachment_id = attachment.id

        try:
            logger.debug(
                f"Downloading content for attachment '{filename}' ({attachment_id})..."
            )
            attachment_content = cast(
                bytes | None, self.confluence_client.download_attachment(attachment_id)
            )
            logger.debug(
                f"Successfully downloaded attachment '{filename}' ({attachment_id})."
            )

            if not attachment_content:
                logger.warning(
                    f"Downloaded content for attachment '{filename}' ({attachment_id}) is empty."
                )
                return None

            return attachment_content

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

    def _parse_attachment_content(
        self, attachment: ConfluenceAttachment, attachment_content: bytes, filename: str
    ) -> str | None:
        """Parse attachment content using the configured parser."""
        attachment_id = attachment.id

        if not self.attachment_parser:
            logger.warning(
                f"Attachment parser not available for attachment '{filename}' ({attachment_id}). Skipping text extraction."
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

    def _parse_html_content(self, html_content: str, page_id: str) -> str | None:
        """Parse HTML content using the configured parser."""
        if not self.html_parser:
            logger.warning(
                f"HTML parser not available for page {page_id}. Skipping text extraction."
            )
            return None

        try:
            logger.debug(f"Parsing HTML content for page {page_id}...")
            extracted_text = self.html_parser.parse(html_content)
            return extracted_text
        except Exception as e:
            logger.error(
                f"Failed to parse HTML content for page {page_id}: {e}",
                exc_info=True,
            )
            return None
