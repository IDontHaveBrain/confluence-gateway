import uuid
from collections.abc import Generator

import pytest
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
from confluence_gateway.core.config import VectorDBConfig
from qdrant_client.http.exceptions import UnexpectedResponse

TEST_DIMENSION = 4
TEST_COLLECTION_NAME = f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="class")
def qdrant_config() -> VectorDBConfig:
    return VectorDBConfig(
        type="qdrant",
        qdrant_url=":memory:",
        collection_name=TEST_COLLECTION_NAME,
        embedding_dimension=TEST_DIMENSION,
    )


@pytest.fixture(scope="class")
def qdrant_adapter(
    qdrant_config: VectorDBConfig,
) -> Generator[QdrantAdapter, None, None]:
    adapter = QdrantAdapter(qdrant_config)
    try:
        adapter.initialize()
        yield adapter
    finally:
        if adapter.client:
            try:
                adapter.client.delete_collection(collection_name=TEST_COLLECTION_NAME)
            except UnexpectedResponse as e:
                if e.status_code != 404:
                    print(f"Error deleting collection during teardown: {e}")
            except Exception as e:
                print(f"Error during adapter teardown: {e}")
            finally:
                adapter.close()


@pytest.fixture(scope="class")
def sample_documents() -> list[Document]:
    return [
        Document(
            id="doc1",
            text="This is the first document.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"space_key": "TEST", "type": "page", "year": 2023},
        ),
        Document(
            id="doc2",
            text="This is the second document, also a page.",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"space_key": "TEST", "type": "page", "year": 2024},
        ),
        Document(
            id="doc3",
            text="This is a blog post.",
            embedding=[0.9, 0.8, 0.7, 0.6],
            metadata={"space_key": "BLOG", "type": "blogpost", "year": 2024},
        ),
    ]


@pytest.mark.integration
class TestQdrantAdapterIntegration:
    def test_initialize(self, qdrant_adapter: QdrantAdapter):
        assert qdrant_adapter.client is not None
        assert qdrant_adapter.config.collection_name == TEST_COLLECTION_NAME
        try:
            collection_info = qdrant_adapter.client.get_collection(
                collection_name=TEST_COLLECTION_NAME
            )
            assert collection_info is not None
            assert collection_info.vectors_config.params.size == TEST_DIMENSION, (
                "Collection dimension mismatch"
            )
        except Exception as e:
            pytest.fail(f"Failed to verify collection existence or dimension: {e}")
        assert qdrant_adapter.count() == 0

    def test_upsert_and_count(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        initial_count = qdrant_adapter.count()
        qdrant_adapter.upsert(sample_documents)
        assert qdrant_adapter.count() == initial_count + len(sample_documents)

        qdrant_adapter.upsert([sample_documents[0]])
        assert qdrant_adapter.count() == initial_count + len(sample_documents)

    def test_search_basic(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        qdrant_adapter.upsert(sample_documents)
        query_embedding = [0.15, 0.25, 0.35, 0.45]
        results = qdrant_adapter.search(query_embedding=query_embedding, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResultItem) for r in results)
        assert results[0].id == "doc1"
        assert 0.0 <= results[0].score <= 1.0
        assert results[0].metadata["space_key"] == "TEST"
        assert results[0].text == sample_documents[0].text

    def test_search_with_filters(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        qdrant_adapter.upsert(sample_documents)
        query_embedding = [0.5, 0.5, 0.5, 0.5]
        filters = {"space_key": "TEST", "year": 2024}
        results = qdrant_adapter.search(
            query_embedding=query_embedding, top_k=3, filters=filters
        )

        assert len(results) == 1
        assert results[0].id == "doc2"
        assert results[0].metadata["space_key"] == "TEST"
        assert results[0].metadata["year"] == 2024

        filters_blog = {"type": "blogpost"}
        results_blog = qdrant_adapter.search(
            query_embedding=query_embedding, top_k=3, filters=filters_blog
        )
        assert len(results_blog) == 1
        assert results_blog[0].id == "doc3"
        assert results_blog[0].metadata["type"] == "blogpost"

    def test_search_by_metadata(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        qdrant_adapter.upsert(sample_documents)

        results_all_test_pages = qdrant_adapter.search_by_metadata(
            filters={"space_key": "TEST", "type": "page"}
        )
        assert len(results_all_test_pages) == 2
        assert {r["id"] for r in results_all_test_pages} == {"doc1", "doc2"}
        assert "year" in results_all_test_pages[0]

        results_blog_select = qdrant_adapter.search_by_metadata(
            filters={"space_key": "BLOG"}, select=["type"], limit=1
        )
        assert len(results_blog_select) == 1
        assert results_blog_select[0]["id"] == "doc3"
        assert list(results_blog_select[0].keys()) == ["id", "type"]
        assert results_blog_select[0]["type"] == "blogpost"

        results_limit_1 = qdrant_adapter.search_by_metadata(
            filters={"type": "page"}, limit=1
        )
        assert len(results_limit_1) == 1

    def test_delete_by_id(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        qdrant_adapter.upsert(sample_documents)
        initial_count = qdrant_adapter.count()
        assert initial_count == len(sample_documents)

        qdrant_adapter.delete(ids=["doc1"])
        assert qdrant_adapter.count() == initial_count - 1

        results_after_delete = qdrant_adapter.search_by_metadata(
            filters={"space_key": "TEST"}
        )
        assert len(results_after_delete) == 1
        assert results_after_delete[0]["id"] == "doc2"

        qdrant_adapter.delete(ids=["non_existent_id"])
        assert qdrant_adapter.count() == initial_count - 1

    def test_delete_by_metadata(
        self, qdrant_adapter: QdrantAdapter, sample_documents: list[Document]
    ):
        qdrant_adapter.upsert(sample_documents)
        initial_count = qdrant_adapter.count()
        assert initial_count == len(sample_documents)

        qdrant_adapter.delete_by_metadata(filters={"space_key": "TEST", "type": "page"})
        assert qdrant_adapter.count() == initial_count - 2

        remaining_docs = qdrant_adapter.search_by_metadata(filters={})
        assert len(remaining_docs) == 1
        assert remaining_docs[0]["id"] == "doc3"

        qdrant_adapter.delete_by_metadata(filters={"year": 2025})
        assert qdrant_adapter.count() == 1

    def test_close(self, qdrant_adapter: QdrantAdapter):
        config = VectorDBConfig(
            type="qdrant",
            qdrant_url=":memory:",
            collection_name=f"close_test_{uuid.uuid4().hex[:8]}",
            embedding_dimension=TEST_DIMENSION,
        )
        adapter_to_close = QdrantAdapter(config)
        adapter_to_close.initialize()
        assert adapter_to_close.client is not None
        adapter_to_close.close()
        assert adapter_to_close.client is None
        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            adapter_to_close.count()
