import io
import logging
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from confluence_gateway.services.parsers.base import ContentParser

# Optional dependency: markitdown
markitdown_module: Optional[ModuleType] = None
MarkItDownClass: Optional[Any] = (
    None  # Use Any temporarily if MarkItDown type causes issues
)
try:
    from markitdown import MarkItDown

    MarkItDownClass = MarkItDown
except ImportError:
    pass  # Keep them None

# Optional dependency: unstructured
unstructured_partition: Optional[Callable[..., list[Any]]] = (
    None  # Use Any for Element type
)
clean_extra_whitespace: Optional[Callable[[str], str]] = None
try:
    from unstructured.cleaners.core import clean_extra_whitespace
    from unstructured.partition.auto import partition as unstructured_partition
except ImportError:
    pass  # Keep them None


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Define types for better static analysis if needed, avoid runtime errors
    pass


class MarkitdownAttachmentParser(ContentParser):
    """Parses attachment content using the markitdown library."""

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
            # markitdown typically works with file paths. Use a temporary file.
            with tempfile.NamedTemporaryFile(
                suffix=f"_{filename}", delete=False
            ) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            # Attempt conversion using the temporary file path
            # Instantiate the converter and call convert
            converter = MarkItDownClass()
            result = converter.convert(tmp_file_path)
            # Basic whitespace cleaning
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
            # Ensure temporary file is deleted
            if tmp_file_path:
                try:
                    Path(tmp_file_path).unlink()
                except OSError:
                    logger.warning(f"Could not delete temporary file: {tmp_file_path}")


class UnstructuredAttachmentParser(ContentParser):
    """Parses attachment content using the unstructured library."""

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
            # Use unstructured's partition function directly with bytes
            elements = unstructured_partition(
                file=io.BytesIO(content),
                file_filename=filename,
                content_type=content_type,
                # Consider adding strategy="fast" or other options if needed
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
