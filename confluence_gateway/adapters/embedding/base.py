from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confluence_gateway.core.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    @abstractmethod
    def __init__(self, config: "EmbeddingConfig") -> None:
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
