from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class ContentParser(ABC):
    """Abstract base class for content parsers (HTML, Attachments)."""

    @abstractmethod
    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]:
        """
        Parses the given content and extracts text.

        Args:
            content: The content to parse (str for HTML, bytes for attachments).
            **kwargs: Additional arguments needed by specific parsers (e.g., filename, content_type).

        Returns:
            The extracted text as a string, or None if parsing fails or yields no text.
        """
        pass
