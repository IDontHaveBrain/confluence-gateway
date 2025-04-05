from abc import ABC, abstractmethod
from typing import Any, Optional

from confluence_gateway.core.config import VectorDBConfig

from .models import Document, VectorSearchResultItem


class VectorDBAdapter(ABC):
    @abstractmethod
    def __init__(self, config: "VectorDBConfig") -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResultItem]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
