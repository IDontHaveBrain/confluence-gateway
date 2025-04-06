from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class ContentParser(ABC):
    @abstractmethod
    def parse(self, content: Union[str, bytes], **kwargs: Any) -> Optional[str]: ...
