from abc import ABC, abstractmethod
from typing import Any

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
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResultItem]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def search_by_metadata(
        self,
        filters: dict[str, Any],
        select: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def delete_by_metadata(self, filters: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
