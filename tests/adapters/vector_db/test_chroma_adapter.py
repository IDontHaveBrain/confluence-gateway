import shutil
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from confluence_gateway.adapters.vector_db.chroma_adapter import ChromaDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.core.config import VectorDBConfig

TEST_DIMENSION = 4
TEST_COLLECTION_NAME = f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="class")
def chroma_persist_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp(f"chroma_test_{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="class")
def chroma_config(chroma_persist_dir: Path) -> VectorDBConfig:
    return VectorDBConfig(
        type="chroma",
        collection_name=TEST_COLLECTION_NAME,
        embedding_dimension=TEST_DIMENSION,
        chroma_persist_path=str(chroma_persist_dir),
    )


@pytest.fixture(scope="class")
def chroma_adapter(
    chroma_config: VectorDBConfig, chroma_persist_dir: Path
) -> Generator[ChromaDBAdapter, None, None]:
    adapter = ChromaDBAdapter(chroma_config)
    try:
        adapter.initialize()
        yield adapter
    finally:
        if adapter.client:
            try:
                adapter.client.delete_collection(name=TEST_COLLECTION_NAME)
            except Exception as e:
                print(f"Error deleting collection during teardown: {e}")
            finally:
                adapter.close()
        if chroma_persist_dir.exists():
            try:
                shutil.rmtree(chroma_persist_dir)
            except Exception as e:
                print(f"Error removing chroma persist directory during teardown: {e}")


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
class TestChromaDBAdapterIntegration:
    def test_initialize(self, chroma_adapter: ChromaDBAdapter):
        assert chroma_adapter.client is not None
        assert chroma_adapter.collection is not None
        assert chroma_adapter.collection.name == TEST_COLLECTION_NAME
        collections = chroma_adapter.client.list_collections()
        assert any(c.name == TEST_COLLECTION_NAME for c in collections)
        assert chroma_adapter.count() == 0

    def test_upsert_and_count(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        initial_count = chroma_adapter.count()
        chroma_adapter.upsert(sample_documents)
        assert chroma_adapter.count() == initial_count + len(sample_documents)
        chroma_adapter.upsert([sample_documents[0]])
        assert chroma_adapter.count() == initial_count + len(sample_documents)

    def test_search_basic(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        chroma_adapter.upsert(sample_documents)
        query_embedding = [0.15, 0.25, 0.35, 0.45]
        results = chroma_adapter.search(query_embedding=query_embedding, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResultItem) for r in results)
        result_ids = {r.id for r in results}
        assert "doc1" in result_ids
        doc1_result = next(r for r in results if r.id == "doc1")
        assert 0.9 <= doc1_result.score <= 1.0
        assert doc1_result.metadata["space_key"] == "TEST"
        assert doc1_result.text == sample_documents[0].text

    def test_search_with_filters(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        chroma_adapter.upsert(sample_documents)
        query_embedding = [0.5, 0.5, 0.5, 0.5]
        filters = {"space_key": "TEST", "year": 2024}
        results = chroma_adapter.search(
            query_embedding=query_embedding, top_k=3, filters=filters
        )

        assert len(results) == 1
        assert results[0].id == "doc2"
        assert results[0].metadata["space_key"] == "TEST"
        assert results[0].metadata["year"] == 2024

        filters_blog = {"type": "blogpost"}
        results_blog = chroma_adapter.search(
            query_embedding=query_embedding, top_k=3, filters=filters_blog
        )
        assert len(results_blog) == 1
        assert results_blog[0].id == "doc3"
        assert results_blog[0].metadata["type"] == "blogpost"

    def test_search_by_metadata(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        chroma_adapter.upsert(sample_documents)

        results_all_test_pages = chroma_adapter.search_by_metadata(
            filters={"space_key": "TEST", "type": "page"}
        )
        assert len(results_all_test_pages) == 2
        assert {r["id"] for r in results_all_test_pages} == {"doc1", "doc2"}
        assert "year" in results_all_test_pages[0]

        results_blog_select = chroma_adapter.search_by_metadata(
            filters={"space_key": "BLOG"}, select=["type"], limit=1
        )
        assert len(results_blog_select) == 1
        assert results_blog_select[0]["id"] == "doc3"
        assert set(results_blog_select[0].keys()) == {"id", "type"}
        assert results_blog_select[0]["type"] == "blogpost"

        results_limit_1 = chroma_adapter.search_by_metadata(
            filters={"type": "page"}, limit=1
        )
        assert len(results_limit_1) == 1

    def test_delete_by_id(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        chroma_adapter.upsert(sample_documents)
        initial_count = chroma_adapter.count()
        assert initial_count == len(sample_documents)

        chroma_adapter.delete(ids=["doc1"])
        assert chroma_adapter.count() == initial_count - 1

        results_after_delete = chroma_adapter.search_by_metadata(
            filters={"space_key": "TEST"}
        )
        assert len(results_after_delete) == 1
        assert results_after_delete[0]["id"] == "doc2"

        chroma_adapter.delete(ids=["non_existent_id"])
        assert chroma_adapter.count() == initial_count - 1

    def test_delete_by_metadata(
        self, chroma_adapter: ChromaDBAdapter, sample_documents: list[Document]
    ):
        chroma_adapter.upsert(sample_documents)
        initial_count = chroma_adapter.count()
        assert initial_count == len(sample_documents)

        chroma_adapter.delete_by_metadata(filters={"space_key": "TEST", "type": "page"})
        assert chroma_adapter.count() == initial_count - 2

        remaining_docs = chroma_adapter.search_by_metadata(filters={})
        assert len(remaining_docs) == 1
        assert remaining_docs[0]["id"] == "doc3"

        chroma_adapter.delete_by_metadata(filters={"year": 2025})
        assert chroma_adapter.count() == 1

    def test_close(self, chroma_config: VectorDBConfig, chroma_persist_dir: Path):
        adapter_to_close = ChromaDBAdapter(chroma_config)
        adapter_to_close.initialize()
        assert adapter_to_close.client is not None
        assert adapter_to_close.collection is not None

        adapter_to_close.close()
        assert adapter_to_close.client is None
        assert adapter_to_close.collection is None

        with pytest.raises(RuntimeError, match="ChromaDB collection not initialized"):
            adapter_to_close.count()

        assert chroma_persist_dir.exists()
