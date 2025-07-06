from abc import ABC, abstractmethod
from typing import Any


class ContentParser(ABC):
    @abstractmethod
    def parse(self, content: str | bytes, **kwargs: Any) -> str | None: ...
