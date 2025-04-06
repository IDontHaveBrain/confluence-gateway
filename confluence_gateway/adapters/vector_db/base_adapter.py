from abc import ABC, abstractmethod
from typing import Any, Optional

from confluence_gateway.core.config import VectorDBConfig

from .models import Document, VectorSearchResultItem


class VectorDBAdapter(ABC):
    @abstractmethod
    def __init__(self, config: "VectorDBConfig") -> None:
        pass

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResultItem]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def search_by_metadata(
        self,
        filters: dict[str, Any],
        select: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete_by_metadata(self, filters: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
