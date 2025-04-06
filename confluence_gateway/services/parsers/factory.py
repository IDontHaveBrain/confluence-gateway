import logging
from typing import Literal

from confluence_gateway.core.exceptions import ConfluenceGatewayError
from confluence_gateway.services.parsers.attachment_parsers import (
    MarkitdownAttachmentParser,
    UnstructuredAttachmentParser,
    unstructured_partition,
)
from confluence_gateway.services.parsers.attachment_parsers import (
    MarkItDownClass as MarkItDownClassAttach,  # Import the class check variable
)
from confluence_gateway.services.parsers.base import ContentParser
from confluence_gateway.services.parsers.html_parsers import (
    MarkItDownClass as MarkItDownClassHtml,  # Import the class check variable
)
from confluence_gateway.services.parsers.html_parsers import (
    MarkitdownHtmlParser,
    UnstructuredHtmlParser,
    partition_html,
)

logger = logging.getLogger(__name__)


class ParserNotAvailableError(ConfluenceGatewayError):
    """Raised when a requested parser cannot be instantiated due to missing dependencies."""

    pass


def get_parser(
    parser_name: str, content_category: Literal["html", "attachment"]
) -> ContentParser:
    """
    Factory function to get a content parser instance based on name and category.

    Args:
        parser_name: The name of the parser (e.g., "markitdown", "unstructured").
        content_category: The type of content ("html" or "attachment").

    Returns:
        An instance of the requested ContentParser.

    Raises:
        ValueError: If the parser_name or content_category is invalid.
        ParserNotAvailableError: If the required library for the parser is not installed.
    """
    parser_name = parser_name.lower()
    logger.debug(
        f"Attempting to get parser: name='{parser_name}', category='{content_category}'"
    )

    if content_category == "html":
        if parser_name == "markitdown":
            if MarkItDownClassHtml is None:  # Check the imported class variable
                raise ParserNotAvailableError(
                    "Markitdown library not installed or MarkItDown class not found. "
                    "Cannot create MarkitdownHtmlParser. Install with 'pip install markitdown'"
                )
            return MarkitdownHtmlParser()
        elif parser_name == "unstructured":
            if partition_html is None:  # Check the imported function variable
                raise ParserNotAvailableError(
                    "Unstructured library (or partition_html function) not installed. "
                    "Cannot create UnstructuredHtmlParser. Install with 'pip install unstructured'"
                )
            return UnstructuredHtmlParser()
        else:
            raise ValueError(
                f"Unsupported HTML parser name: '{parser_name}'. Must be 'markitdown' or 'unstructured'."
            )

    elif content_category == "attachment":
        if parser_name == "markitdown":
            if MarkItDownClassAttach is None:  # Check the imported class variable
                raise ParserNotAvailableError(
                    "Markitdown library not installed or MarkItDown class not found. "
                    "Cannot create MarkitdownAttachmentParser. Install with 'pip install markitdown[cli]'"
                )
            return MarkitdownAttachmentParser()
        elif parser_name == "unstructured":
            if unstructured_partition is None:  # Check the imported function variable
                raise ParserNotAvailableError(
                    "Unstructured library (or partition function) not installed. "
                    "Cannot create UnstructuredAttachmentParser. Install with 'pip install unstructured[local-inference]'"
                )
            return UnstructuredAttachmentParser()
        else:
            raise ValueError(
                f"Unsupported attachment parser name: '{parser_name}'. Must be 'markitdown' or 'unstructured'."
            )

    else:
        raise ValueError(
            f"Invalid content category: '{content_category}'. Must be 'html' or 'attachment'."
        )
