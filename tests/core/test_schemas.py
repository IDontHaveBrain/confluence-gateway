from datetime import datetime

import pytest
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.api.schemas.requests import (
    AdvancedSearchRequest,
    BaseSearchRequest,
    CQLSearchRequest,
    GenerateAnswerRequest,
    IndexingTriggerRequest,
    SemanticSearchRequest,
    TextSearchRequest,
)
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    GenerateAnswerResponse,
    IndexingStatusResponse,
    SearchResponse,
    SearchResultItem,
    SemanticSearchResponse,
    SourceDocument,
)
from confluence_gateway.core.config import search_config
from pydantic import ValidationError


pytestmark = pytest.mark.unit


def test_base_search_request_validation():
    BaseSearchRequest(limit=10, start=0)
    BaseSearchRequest(limit=search_config.max_limit, start=100)
    BaseSearchRequest()

    with pytest.raises(ValidationError, match="Limit must be a positive integer"):
        BaseSearchRequest(limit=0)
    with pytest.raises(ValidationError, match="Limit must be a positive integer"):
        BaseSearchRequest(limit=-1)
    with pytest.raises(
        ValidationError, match=f"Limit cannot exceed {search_config.max_limit}"
    ):
        BaseSearchRequest(limit=search_config.max_limit + 1)

    with pytest.raises(ValidationError, match="Start position cannot be negative"):
        BaseSearchRequest(start=-1)


def test_text_search_request_validation():
    TextSearchRequest(query="test query")
    TextSearchRequest(query="ok", content_type="page", space_key="DEV")
    TextSearchRequest(query="ok", content_type="attachment")

    with pytest.raises(ValidationError, match="Query must be at least 2 characters"):
        TextSearchRequest(query="a")
    with pytest.raises(ValidationError, match="Query must be at least 2 characters"):
        TextSearchRequest(query=" ")
    with pytest.raises(ValidationError):
        TextSearchRequest()

    with pytest.raises(ValidationError, match="Invalid content type"):
        TextSearchRequest(query="ok", content_type="invalid_type")


def test_cql_search_request_validation():
    CQLSearchRequest(cql="space = DEV and type = page")
    CQLSearchRequest(cql="title ~ 'test'")

    with pytest.raises(ValidationError, match="CQL query cannot be empty"):
        CQLSearchRequest(cql="")
    with pytest.raises(ValidationError, match="CQL query cannot be empty"):
        CQLSearchRequest(cql=" ")
    with pytest.raises(ValidationError, match="Invalid CQL query format"):
        CQLSearchRequest(cql="just text")
    with pytest.raises(ValidationError):
        CQLSearchRequest()


def test_advanced_search_request_validation():
    AdvancedSearchRequest(query="test")
    AdvancedSearchRequest(
        query="test",
        min_relevance=0.5,
        top_n=10,
        sort_by=["updated_at", "title"],
        sort_direction=["desc", "asc"],
    )
    AdvancedSearchRequest(query="test", get_all_results=True, max_results=500)

    with pytest.raises(ValidationError, match="Query must be at least 2 characters"):
        AdvancedSearchRequest(query="a")

    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        AdvancedSearchRequest(query="test", min_relevance=-0.1)
    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 1"
    ):
        AdvancedSearchRequest(query="test", min_relevance=1.1)

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        AdvancedSearchRequest(query="test", top_n=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        AdvancedSearchRequest(query="test", top_n=-1)

    with pytest.raises(ValidationError, match="Invalid sort field"):
        AdvancedSearchRequest(query="test", sort_by=["invalid_field"])

    with pytest.raises(ValidationError, match="Invalid sort direction"):
        AdvancedSearchRequest(query="test", sort_direction=["up"])

    with pytest.raises(ValidationError, match="max_results must be a positive integer"):
        AdvancedSearchRequest(query="test", get_all_results=True, max_results=0)
    with pytest.raises(
        ValidationError,
        match="max_results can only be used when get_all_results is True",
    ):
        AdvancedSearchRequest(query="test", get_all_results=False, max_results=100)


def test_semantic_search_request_validation():
    SemanticSearchRequest(query="semantic query")
    SemanticSearchRequest(query="ok", top_k=5, filters={"space_key": "DEV"})

    with pytest.raises(ValidationError, match="Query must be at least 2 characters"):
        SemanticSearchRequest(query="a")
    with pytest.raises(ValidationError):
        SemanticSearchRequest()

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        SemanticSearchRequest(query="ok", top_k=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        SemanticSearchRequest(query="ok", top_k=-5)


def test_generate_answer_request_validation():
    GenerateAnswerRequest(query="generate answer")
    GenerateAnswerRequest(query="ok", top_k_retrieval=3, filters={"label": "api"})

    with pytest.raises(ValidationError, match="Query must be at least 2 characters"):
        GenerateAnswerRequest(query="a")
    with pytest.raises(ValidationError):
        GenerateAnswerRequest()

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        GenerateAnswerRequest(query="ok", top_k_retrieval=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        GenerateAnswerRequest(query="ok", top_k_retrieval=-1)


def test_indexing_trigger_request_validation():
    IndexingTriggerRequest()
    IndexingTriggerRequest(space_keys=["DEV", "PROD"])
    IndexingTriggerRequest(space_keys=[])
    IndexingTriggerRequest(space_keys=None)

    with pytest.raises(ValidationError):
        IndexingTriggerRequest(space_keys="DEV")


def test_search_result_item_instantiation():
    now = datetime.now()
    item = SearchResultItem(
        id="123",
        title="Test Page",
        type="page",
        space_key="DEV",
        space_name="Development",
        url="http://example.com",
        last_modified=now,
        excerpt="An excerpt",
    )
    assert item.id == "123"
    assert item.last_modified == now


def test_search_response_instantiation():
    now = datetime.now()
    item = SearchResultItem(
        id="123",
        title="Test",
        type="page",
        space_key="T",
        space_name="Test",
        url="http://ex",
        last_modified=now,
    )
    resp = SearchResponse(
        results=[item],
        total=1,
        start=0,
        limit=10,
        took_ms=123.45,
        page_count=1,
        current_page=1,
        has_more=False,
    )
    assert len(resp.results) == 1
    assert resp.total == 1
    assert resp.took_ms == 123.45


def test_indexing_status_response_instantiation():
    now = datetime.now()
    resp = IndexingStatusResponse(
        status="success",
        last_run_start_time=now,
        last_run_end_time=now,
        last_error_message=None,
    )
    assert resp.status == "success"
    assert resp.last_run_start_time == now


def test_error_response_instantiation():
    resp = ErrorResponse(code=400, message="Bad Request", details={"field": "query"})
    assert resp.status == "error"
    assert resp.code == 400
    assert resp.details["field"] == "query"


def test_semantic_search_response_instantiation():
    item = VectorSearchResultItem(
        id="chunk_1", score=0.9, metadata={"title": "Semantic Doc"}, text="Some text"
    )
    resp = SemanticSearchResponse(results=[item], took_ms=50.1, query="semantic query")
    assert len(resp.results) == 1
    assert resp.results[0].score == 0.9
    assert resp.query == "semantic query"


def test_source_document_from_vector_search_item():
    vector_item_full = VectorSearchResultItem(
        id="vec_1",
        score=0.85,
        metadata={
            "title": "Source Title",
            "url": "http://source.url",
            "space_key": "SRC",
            "other_meta": "value",
        },
    )
    source_doc = SourceDocument.from_vector_search_item(vector_item_full)
    assert source_doc.id == "vec_1"
    assert source_doc.score == 0.85
    assert source_doc.title == "Source Title"
    assert source_doc.url == "http://source.url"
    assert source_doc.space_key == "SRC"

    vector_item_partial = VectorSearchResultItem(
        id="vec_2", score=0.7, metadata={"title": "Partial"}
    )
    source_doc_partial = SourceDocument.from_vector_search_item(vector_item_partial)
    assert source_doc_partial.id == "vec_2"
    assert source_doc_partial.score == 0.7
    assert source_doc_partial.title == "Partial"
    assert source_doc_partial.url is None
    assert source_doc_partial.space_key is None

    vector_item_no_meta = VectorSearchResultItem(id="vec_3", score=0.6, metadata={})
    source_doc_no_meta = SourceDocument.from_vector_search_item(vector_item_no_meta)
    assert source_doc_no_meta.title is None
    assert source_doc_no_meta.url is None
    assert source_doc_no_meta.space_key is None


def test_generate_answer_response_instantiation():
    source = SourceDocument(
        id="src_1", score=0.9, title="Source Doc", url="http://src", space_key="DOC"
    )
    resp = GenerateAnswerResponse(
        answer="This is the generated answer.", sources=[source]
    )
    assert "generated answer" in resp.answer
    assert len(resp.sources) == 1
    assert resp.sources[0].id == "src_1"
