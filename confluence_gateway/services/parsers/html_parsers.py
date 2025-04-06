import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from confluence_gateway.services.parsers.base import ContentParser

# Optional dependency: markitdown
markitdown_module: Optional[ModuleType] = None
MarkItDownClass: Optional[Any] = None  # Use Any temporarily
try:
    from markitdown import MarkItDown

    MarkItDownClass = MarkItDown
except ImportError:
    pass  # Keep them None

# Optional dependency: unstructured
partition_html: Optional[Callable[..., list[Any]]] = None  # Use Any for Element type
clean_extra_whitespace: Optional[Callable[[str], str]] = None
try:
    from unstructured.cleaners.core import clean_extra_whitespace
    from unstructured.partition.html import partition_html
except ImportError:
    pass  # Keep them None


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Define types for better static analysis if needed
    pass


class MarkitdownHtmlParser(ContentParser):
    """Parses HTML content using the markitdown library."""

    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]:
        if MarkItDownClass is None:
            logger.error(
                "Markitdown library not installed or MarkItDown class not found. "
                "Cannot use MarkitdownHtmlParser. Install with 'pip install markitdown'"
            )
            return None

        if not isinstance(content, str) or not content:
            logger.debug("MarkitdownHtmlParser received empty or non-string content.")
            return None

        try:
            # Instantiate the converter and call convert
            converter = MarkItDownClass()
            result = converter.convert(content)
            # Basic whitespace cleaning
            extracted_text = " ".join(result.markdown.split())
            logger.debug("Successfully extracted text using markitdown")
            return extracted_text if extracted_text else None
        except Exception as e:
            logger.error(f"Markitdown failed to parse HTML content: {e}", exc_info=True)
            return None


class UnstructuredHtmlParser(ContentParser):
    """Parses HTML content using the unstructured library."""

    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]:
        if partition_html is None:
            logger.error(
                "Unstructured library not installed or partition_html not found. "
                "Cannot use UnstructuredHtmlParser. Install with 'pip install unstructured'"
            )
            return None

        if not isinstance(content, str) or not content:
            logger.debug("UnstructuredHtmlParser received empty or non-string content.")
            return None

        try:
            elements = partition_html(text=content)
            combined_text = "\n\n".join(
                [el.text for el in elements if hasattr(el, "text")]
            )

            if clean_extra_whitespace is not None:
                combined_text = clean_extra_whitespace(combined_text)

            logger.debug(
                f"Successfully extracted text using unstructured (combined {len(elements)} elements)"
            )
            return combined_text if combined_text else None
        except Exception as e:
            logger.error(
                f"Unstructured failed to parse HTML content: {e}", exc_info=True
            )
            return None
