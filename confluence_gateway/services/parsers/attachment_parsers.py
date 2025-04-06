import io
import logging
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from confluence_gateway.services.parsers.base import ContentParser

markitdown_module: Optional[ModuleType] = None
MarkItDownClass: Optional[Any] = None
try:
    from markitdown import MarkItDown

    MarkItDownClass = MarkItDown
except ImportError:
    MarkItDownClass = None

unstructured_partition: Optional[Callable[..., list[Any]]] = None
clean_extra_whitespace: Optional[Callable[[str], str]] = None
try:
    from unstructured.cleaners.core import clean_extra_whitespace
    from unstructured.partition.auto import partition as unstructured_partition
except ImportError:
    unstructured_partition = None
    clean_extra_whitespace = None


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass


class MarkitdownAttachmentParser(ContentParser):
    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]:
        if MarkItDownClass is None:
            logger.error(
                "Markitdown library not installed or MarkItDown class not found. "
                "Cannot use MarkitdownAttachmentParser. Install with 'pip install markitdown[cli]'"
            )
            return None

        if not isinstance(content, bytes) or not content:
            logger.debug(
                "MarkitdownAttachmentParser received empty or non-bytes content."
            )
            return None

        filename = kwargs.get("filename", "unknown_attachment")
        tmp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=f"_{filename}", delete=False
            ) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            converter = MarkItDownClass()
            result = converter.convert(tmp_file_path)
            extracted_text = " ".join(result.markdown.split())
            logger.debug(
                f"Successfully extracted text from attachment '{filename}' using markitdown"
            )
            return extracted_text if extracted_text else None
        except Exception as e:
            logger.error(
                f"Markitdown failed to parse attachment '{filename}': {e}",
                exc_info=True,
            )
            return None
        finally:
            if tmp_file_path:
                try:
                    Path(tmp_file_path).unlink()
                except OSError:
                    logger.warning(f"Could not delete temporary file: {tmp_file_path}")


class UnstructuredAttachmentParser(ContentParser):
    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]:
        if unstructured_partition is None:
            logger.error(
                "Unstructured library not installed or partition function not found. "
                "Cannot use UnstructuredAttachmentParser. Install with 'pip install unstructured[local-inference]'"
            )
            return None

        if not isinstance(content, bytes) or not content:
            logger.debug(
                "UnstructuredAttachmentParser received empty or non-bytes content."
            )
            return None

        filename = kwargs.get("filename", "unknown_attachment")
        content_type = kwargs.get("content_type")

        try:
            elements = unstructured_partition(
                file=io.BytesIO(content),
                file_filename=filename,
                content_type=content_type,
            )
            combined_text = "\n\n".join(
                [el.text for el in elements if hasattr(el, "text")]
            )

            if clean_extra_whitespace is not None:
                combined_text = clean_extra_whitespace(combined_text)

            logger.debug(
                f"Successfully extracted text from attachment '{filename}' using unstructured (combined {len(elements)} elements)"
            )
            return combined_text if combined_text else None
        except Exception as e:
            logger.error(
                f"Unstructured failed to parse attachment '{filename}': {e}",
                exc_info=True,
            )
            return None
