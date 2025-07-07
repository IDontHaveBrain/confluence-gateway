import io
import logging
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from confluence_gateway.services.parsers.base import ContentParser

markitdown_module: ModuleType | None = None
MarkItDownClass: Any | None = None
try:
    from markitdown import MarkItDown
    from markitdown._stream_info import StreamInfo

    MarkItDownClass = MarkItDown
except ImportError:
    MarkItDownClass = None


partition_html: Callable[..., list[Any]] | None = None
clean_extra_whitespace: Callable[[str], str] | None = None
try:
    from unstructured.cleaners.core import (
        clean_extra_whitespace as _clean_extra_whitespace,
    )
    from unstructured.partition.html import partition_html as _partition_html

    partition_html = _partition_html
    clean_extra_whitespace = _clean_extra_whitespace
except ImportError:
    partition_html = None
    clean_extra_whitespace = None


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass


class MarkitdownHtmlParser(ContentParser):
    def parse(self, content: str | bytes, **kwargs: Any) -> str | None:
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
            converter = MarkItDownClass()
            html_bytes = content.encode("utf-8")
            html_stream = io.BytesIO(html_bytes)
            stream_info = StreamInfo(mimetype="text/html", charset="utf-8")
            result = converter.convert(html_stream, stream_info=stream_info)
            extracted_text = " ".join(result.markdown.split())
            logger.debug("Successfully extracted text using markitdown")
            return extracted_text if extracted_text else None
        except Exception as e:
            logger.error(f"Markitdown failed to parse HTML content: {e}", exc_info=True)
            return None


class UnstructuredHtmlParser(ContentParser):
    def parse(self, content: str | bytes, **kwargs: Any) -> str | None:
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
