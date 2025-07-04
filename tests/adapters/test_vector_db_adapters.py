import time
import uuid

import pytest
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.adapters.vector_db.chroma_adapter import ChromaDBAdapter
from confluence_gateway.adapters.vector_db.models import (
    Document,
    VectorSearchResultItem,
)
from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
from confluence_gateway.services.embedding import EmbeddingService

pytestmark = [pytest.mark.integration, pytest.mark.semantic]


@pytest.fixture(scope="module")
def adapter(vector_db_adapter: VectorDBAdapter) -> VectorDBAdapter:
    if not vector_db_adapter:
        pytest.skip("Vector DB adapter fixture not available.")
    print(f"\nINFO: Using Vector DB Adapter: {type(vector_db_adapter).__name__}")
    return vector_db_adapter


@pytest.fixture(scope="module")
def embed_svc(embedding_service: EmbeddingService) -> EmbeddingService:
    if not embedding_service:
        pytest.skip("Embedding service fixture not available.")
    return embedding_service


def create_test_doc(
    embed_svc: EmbeddingService, text: str, metadata: dict, prefix: str = "test"
) -> Document:
    doc_id = str(uuid.uuid4())
    embedding = embed_svc.embed_text(text)
    return Document(id=doc_id, text=text, embedding=embedding, metadata=metadata)


def test_adapter_initialization(adapter: VectorDBAdapter):
    assert adapter is not None
    if isinstance(adapter, QdrantAdapter):
        assert adapter.client is not None
    elif isinstance(adapter, ChromaDBAdapter):
        assert adapter.client is not None
        assert adapter.collection is not None


def test_adapter_count_initial(
    adapter: VectorDBAdapter, SEMANTIC_TEST_DOCS: list[dict], index_semantic_test_data
):
    initial_count = adapter.count()
    print(f"\nINFO: Initial document count: {initial_count}")
    assert initial_count >= len(SEMANTIC_TEST_DOCS)


def test_adapter_upsert_and_count(
    adapter: VectorDBAdapter, embed_svc: EmbeddingService
):
    initial_count = adapter.count()
    docs_to_add = [
        create_test_doc(
            embed_svc,
            "Unique text one for upsert test.",
            {"source": "upsert_test", "num": 1},
        ),
        create_test_doc(
            embed_svc,
            "Unique text two for upsert test.",
            {"source": "upsert_test", "num": 2},
        ),
    ]
    adapter.upsert(docs_to_add)
    time.sleep(0.5)
    new_count = adapter.count()
    assert new_count == initial_count + len(docs_to_add)

    docs_to_update = [
        Document(
            id=docs_to_add[0].id,
            text="Updated text one.",
            embedding=embed_svc.embed_text("Updated text one."),
            metadata={"source": "upsert_test", "num": 1, "updated": True},
        )
    ]
    adapter.upsert(docs_to_update)
    time.sleep(0.5)
    final_count = adapter.count()
    assert final_count == new_count


def test_adapter_search(
    adapter: VectorDBAdapter,
    embed_svc: EmbeddingService,
    SEMANTIC_TEST_DOCS: list[dict],
    index_semantic_test_data,
):
    search_text = "apples"
    query_embedding = embed_svc.embed_text(search_text)

    results = adapter.search(query_embedding=query_embedding, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, VectorSearchResultItem) for r in results)
    assert results[0].id == SEMANTIC_TEST_DOCS[0]["id"]
    assert isinstance(results[0].score, float)
    assert 0.0 <= results[0].score <= 1.0
    assert "text" in results[0].metadata or results[0].text is not None


def test_adapter_search_with_filter(
    adapter: VectorDBAdapter, embed_svc: EmbeddingService
):
    filter_docs = [
        create_test_doc(
            embed_svc,
            "Document for filtering test - category A.",
            {"category": "A", "test_id": "filter_test"},
        ),
        create_test_doc(
            embed_svc,
            "Another document for filtering - category B.",
            {"category": "B", "test_id": "filter_test"},
        ),
    ]
    adapter.upsert(filter_docs)
    time.sleep(0.5)

    search_text = "filtering test"
    query_embedding = embed_svc.embed_text(search_text)

    results_a = adapter.search(
        query_embedding=query_embedding, top_k=5, filters={"category": "A"}
    )
    assert len(results_a) >= 1
    assert all(r.metadata.get("category") == "A" for r in results_a if r.metadata)
    assert any(r.id == filter_docs[0].id for r in results_a)
    assert not any(r.id == filter_docs[1].id for r in results_a)

    results_b = adapter.search(
        query_embedding=query_embedding, top_k=5, filters={"category": "B"}
    )
    assert len(results_b) >= 1
    assert all(r.metadata.get("category") == "B" for r in results_b if r.metadata)
    assert any(r.id == filter_docs[1].id for r in results_b)
    assert not any(r.id == filter_docs[0].id for r in results_b)


def test_adapter_search_by_metadata(
    adapter: VectorDBAdapter, embed_svc: EmbeddingService
):
    meta_docs = [
        create_test_doc(
            embed_svc, "Metadata search doc 1", {"search_key": "meta_test", "value": 1}
        ),
        create_test_doc(
            embed_svc, "Metadata search doc 2", {"search_key": "meta_test", "value": 2}
        ),
        create_test_doc(
            embed_svc, "Other metadata doc", {"search_key": "other", "value": 3}
        ),
    ]
    adapter.upsert(meta_docs)
    time.sleep(0.5)

    results = adapter.search_by_metadata(filters={"search_key": "meta_test"})
    assert len(results) == 2
    result_ids = {r["id"] for r in results}
    assert meta_docs[0].id in result_ids
    assert meta_docs[1].id in result_ids
    assert meta_docs[2].id not in result_ids
    assert "search_key" in results[0]
    assert "value" in results[0]

    results_limit = adapter.search_by_metadata(
        filters={"search_key": "meta_test"}, limit=1
    )
    assert len(results_limit) == 1

    select_key = "value"
    f"metadata.{select_key}" if isinstance(adapter, QdrantAdapter) else select_key
    results_select = adapter.search_by_metadata(
        filters={"search_key": "meta_test"}, select=[select_key], limit=1
    )
    assert len(results_select) == 1
    assert list(results_select[0].keys()) == ["id", select_key]


@pytest.mark.skipif(
    not hasattr(pytest.lazy_fixture("vector_db_adapter"), "retrieve_by_ids"),
    reason="Adapter does not support retrieve_by_ids",
)
def test_adapter_retrieve_by_ids(adapter: VectorDBAdapter, embed_svc: EmbeddingService):
    doc_to_retrieve = create_test_doc(
        embed_svc, "Document to retrieve by ID.", {"source": "retrieve_test"}
    )
    adapter.upsert([doc_to_retrieve])
    time.sleep(0.5)

    retrieved = adapter.retrieve_by_ids(ids=[doc_to_retrieve.id])
    assert len(retrieved) == 1
    assert str(retrieved[0].id) == doc_to_retrieve.id
    assert retrieved[0].payload["metadata"]["source"] == "retrieve_test"

    retrieved_non_existent = adapter.retrieve_by_ids(ids=["non_existent_id"])
    assert len(retrieved_non_existent) == 0


def test_adapter_delete_by_id(adapter: VectorDBAdapter, embed_svc: EmbeddingService):
    doc_to_delete = create_test_doc(
        embed_svc, "Document to be deleted by ID.", {"source": "delete_id_test"}
    )
    adapter.upsert([doc_to_delete])
    time.sleep(0.5)

    count_before = adapter.count()
    adapter.delete(ids=[doc_to_delete.id])
    time.sleep(0.5)
    count_after = adapter.count()

    assert count_after == count_before - 1

    results = adapter.search_by_metadata(filters={"source": "delete_id_test"})
    assert len(results) == 0


def test_adapter_delete_by_metadata(
    adapter: VectorDBAdapter, embed_svc: EmbeddingService
):
    delete_meta_docs = [
        create_test_doc(
            embed_svc, "Metadata delete doc 1", {"delete_key": "meta_del", "num": 1}
        ),
        create_test_doc(
            embed_svc, "Metadata delete doc 2", {"delete_key": "meta_del", "num": 2}
        ),
        create_test_doc(embed_svc, "Keep this doc", {"delete_key": "keep", "num": 3}),
    ]
    adapter.upsert(delete_meta_docs)
    time.sleep(0.5)

    count_before = adapter.count()
    adapter.delete_by_metadata(filters={"delete_key": "meta_del"})
    time.sleep(0.5)
    count_after = adapter.count()

    assert count_after == count_before - 2

    results_deleted = adapter.search_by_metadata(filters={"delete_key": "meta_del"})
    assert len(results_deleted) == 0
    results_kept = adapter.search_by_metadata(filters={"delete_key": "keep"})
    assert len(results_kept) == 1
    assert results_kept[0]["id"] == delete_meta_docs[2].id


def test_adapter_close(adapter: VectorDBAdapter):
    adapter_type = type(adapter)
    config = adapter.config

    temp_adapter: VectorDBAdapter
    if isinstance(adapter, QdrantAdapter):
        temp_adapter = QdrantAdapter(config)
    elif isinstance(adapter, ChromaDBAdapter):
        temp_adapter = ChromaDBAdapter(config)
    else:
        pytest.skip(f"Close test not implemented for adapter type {adapter_type}")
        return

    temp_adapter.initialize()
    temp_adapter.close()

    if isinstance(temp_adapter, QdrantAdapter):
        assert temp_adapter.client is None
    elif isinstance(temp_adapter, ChromaDBAdapter):
        assert temp_adapter.client is None
        assert temp_adapter.collection is None
