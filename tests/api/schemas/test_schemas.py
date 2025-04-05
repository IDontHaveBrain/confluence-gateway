from datetime import datetime

import pytest
from confluence_gateway.api.schemas.requests import (
    AdvancedSearchRequest,
    BaseSearchRequest,
    CQLSearchRequest,
    TextSearchRequest,
)
from confluence_gateway.api.schemas.responses import (
    ErrorResponse,
    PaginationLinks,
    SearchResponse,
    SearchResultItem,
)
from pydantic import ValidationError


class TestBaseSearchRequest:
    def test_valid_base_search_request(self):
        request = BaseSearchRequest(limit=10, start=0, expand=["body.view", "space"])
        assert request.limit == 10
        assert request.start == 0
        assert request.expand == ["body.view", "space"]

        request = BaseSearchRequest()
        assert request.limit is None
        assert request.start is None
        assert request.expand is None

    def test_invalid_limit(self):
        with pytest.raises(ValidationError) as e:
            BaseSearchRequest(limit=-5)
        assert "Limit must be a positive integer" in str(e.value)

        with pytest.raises(ValidationError) as e:
            BaseSearchRequest(limit=0)
        assert "Limit must be a positive integer" in str(e.value)

        with pytest.raises(ValidationError) as e:
            BaseSearchRequest(limit=101)
        assert "Limit cannot exceed" in str(e.value)

    def test_invalid_start(self):
        with pytest.raises(ValidationError) as e:
            BaseSearchRequest(start=-1)
        assert "Start position cannot be negative" in str(e.value)


class TestTextSearchRequest:
    def test_valid_text_search_request(self):
        request = TextSearchRequest(
            query="test query",
            space_key="DEV",
            content_type="page",
            include_archived=False,
            limit=20,
            start=0,
        )
        assert request.query == "test query"
        assert request.space_key == "DEV"
        assert request.content_type == "page"
        assert request.include_archived is False
        assert request.limit == 20
        assert request.start == 0

        request = TextSearchRequest(query="test query")
        assert request.query == "test query"
        assert request.space_key is None
        assert request.content_type is None
        assert request.include_archived is False
        assert request.limit is None
        assert request.start is None

    def test_invalid_query(self):
        with pytest.raises(ValidationError) as e:
            TextSearchRequest(query="")
        assert "Query must be at least 2 characters long" in str(e.value)

        with pytest.raises(ValidationError) as e:
            TextSearchRequest(query="a")
        assert "Query must be at least 2 characters long" in str(e.value)

        with pytest.raises(ValidationError) as e:
            TextSearchRequest(query="  ")
        assert "Query must be at least 2 characters long" in str(e.value)

    def test_invalid_content_type(self):
        with pytest.raises(ValidationError) as e:
            TextSearchRequest(query="test query", content_type="invalid_type")
        assert "Invalid content type" in str(e.value)

        for valid_type in ["page", "blogpost", "attachment", "comment"]:
            request = TextSearchRequest(query="test query", content_type=valid_type)
            assert request.content_type == valid_type


class TestCQLSearchRequest:
    def test_valid_cql_search_request(self):
        request = CQLSearchRequest(cql="space = DEV AND type = page", limit=20, start=0)
        assert request.cql == "space = DEV AND type = page"
        assert request.limit == 20
        assert request.start == 0

        complex_cql = (
            "space = DEV AND (type = page OR type = blogpost) AND text ~ 'api'"
        )
        request = CQLSearchRequest(cql=complex_cql)
        assert request.cql == complex_cql

    def test_invalid_cql(self):
        with pytest.raises(ValidationError) as e:
            CQLSearchRequest(cql="")
        assert "CQL query cannot be empty" in str(e.value)

        with pytest.raises(ValidationError) as e:
            CQLSearchRequest(cql="  ")
        assert "CQL query cannot be empty" in str(e.value)

        with pytest.raises(ValidationError) as e:
            CQLSearchRequest(cql="just some text")
        assert "Invalid CQL query format" in str(e.value)


class TestAdvancedSearchRequest:
    def test_valid_advanced_search_request(self):
        request = AdvancedSearchRequest(
            query="api documentation",
            space_key="DEV",
            content_type="page",
            include_archived=False,
            get_all_results=True,
            max_results=100,
            min_relevance=0.5,
            top_n=10,
            sort_by=["updated_at", "title"],
            sort_direction=["desc", "asc"],
            limit=20,
            start=0,
        )
        assert request.query == "api documentation"
        assert request.space_key == "DEV"
        assert request.content_type == "page"
        assert request.include_archived is False
        assert request.get_all_results is True
        assert request.max_results == 100
        assert request.min_relevance == 0.5
        assert request.top_n == 10
        assert request.sort_by == ["updated_at", "title"]
        assert request.sort_direction == ["desc", "asc"]
        assert request.limit == 20
        assert request.start == 0

        request = AdvancedSearchRequest(query="api documentation")
        assert request.query == "api documentation"
        assert request.space_key is None
        assert request.content_type is None
        assert request.include_archived is False
        assert request.get_all_results is False
        assert request.max_results is None
        assert request.min_relevance is None
        assert request.top_n is None
        assert request.sort_by is None
        assert request.sort_direction is None
        assert request.limit is None
        assert request.start is None

    def test_invalid_query(self):
        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="")
        assert "Query must be at least 2 characters long" in str(e.value)

    def test_invalid_content_type(self):
        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="test", content_type="invalid_type")
        assert "Invalid content type" in str(e.value)

    def test_invalid_sort_fields(self):
        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="test", sort_by=["invalid_field"])
        assert "Invalid sort field" in str(e.value)

        valid_fields = ["title", "created_at", "updated_at", "score", "space_key"]
        for field in valid_fields:
            request = AdvancedSearchRequest(query="test", sort_by=[field])
            assert request.sort_by == [field]

    def test_invalid_sort_direction(self):
        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="test", sort_direction=["invalid"])
        assert "Invalid sort direction" in str(e.value)

        for direction in ["asc", "desc"]:
            request = AdvancedSearchRequest(query="test", sort_direction=[direction])
            assert request.sort_direction == [direction]

    def test_max_results_validation(self):
        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="test", max_results=50, get_all_results=False)
        assert "max_results can only be used when get_all_results is True" in str(
            e.value
        )

        with pytest.raises(ValidationError) as e:
            AdvancedSearchRequest(query="test", max_results=-1, get_all_results=True)
        assert "max_results must be a positive integer" in str(e.value)

        request = AdvancedSearchRequest(
            query="test", max_results=50, get_all_results=True
        )
        assert request.max_results == 50
        assert request.get_all_results is True


class TestSearchResponseSchemas:
    def test_pagination_links(self):
        links = PaginationLinks(
            next="/api/search?query=api&start=20&limit=20",
            previous="/api/search?query=api&start=0&limit=20",
        )
        assert links.next == "/api/search?query=api&start=20&limit=20"
        assert links.previous == "/api/search?query=api&start=0&limit=20"

        links = PaginationLinks(next="/api/search?query=api&start=20&limit=20")
        assert links.next == "/api/search?query=api&start=20&limit=20"
        assert links.previous is None

        links = PaginationLinks()
        assert links.next is None
        assert links.previous is None

    def test_search_result_item(self):
        item = SearchResultItem(
            id="12345",
            title="API Documentation",
            type="page",
            space_key="DEV",
            space_name="Development",
            url="https://confluence.example.com/display/DEV/API+Documentation",
            last_modified=datetime.fromisoformat("2023-05-15T14:32:21"),
        )
        assert item.id == "12345"
        assert item.title == "API Documentation"
        assert item.type == "page"
        assert item.space_key == "DEV"
        assert item.space_name == "Development"
        assert (
            item.url == "https://confluence.example.com/display/DEV/API+Documentation"
        )
        assert item.excerpt is None
        assert item.last_modified == datetime.fromisoformat("2023-05-15T14:32:21")

        item = SearchResultItem(
            id="12345",
            title="API Documentation",
            type="page",
            space_key="DEV",
            space_name="Development",
            url="https://confluence.example.com/display/DEV/API+Documentation",
            excerpt="This document describes the <em>API</em> endpoints...",
            last_modified=datetime.fromisoformat("2023-05-15T14:32:21"),
        )
        assert item.excerpt == "This document describes the <em>API</em> endpoints..."

    def test_search_response(self):
        response = SearchResponse(
            total=42,
            start=0,
            limit=20,
            took_ms=123.45,
            page_count=3,
            current_page=1,
            has_more=True,
        )
        assert response.total == 42
        assert response.start == 0
        assert response.limit == 20
        assert response.took_ms == 123.45
        assert response.page_count == 3
        assert response.current_page == 1
        assert response.has_more is True
        assert response.results == []
        assert response.links is None

        item = SearchResultItem(
            id="12345",
            title="API Documentation",
            type="page",
            space_key="DEV",
            space_name="Development",
            url="https://confluence.example.com/display/DEV/API+Documentation",
            last_modified=datetime.fromisoformat("2023-05-15T14:32:21"),
        )
        links = PaginationLinks(
            next="/api/search?query=api&start=20&limit=20", previous=None
        )
        response = SearchResponse(
            results=[item],
            total=42,
            start=0,
            limit=20,
            took_ms=123.45,
            page_count=3,
            current_page=1,
            has_more=True,
            links=links,
        )
        assert len(response.results) == 1
        assert response.results[0].id == "12345"
        assert response.links.next == "/api/search?query=api&start=20&limit=20"
        assert response.links.previous is None

    def test_error_response(self):
        error = ErrorResponse(
            code=400,
            message="Invalid search parameters",
        )
        assert error.status == "error"
        assert error.code == 400
        assert error.message == "Invalid search parameters"
        assert error.details is None

        error = ErrorResponse(
            status="error",
            code=400,
            message="Invalid search parameters",
            details={
                "param": "query",
                "reason": "Query must be at least 2 characters long",
            },
        )
        assert error.status == "error"
        assert error.code == 400
        assert error.message == "Invalid search parameters"
        assert error.details["param"] == "query"
        assert error.details["reason"] == "Query must be at least 2 characters long"


class TestJSONSerialization:
    def test_request_serialization(self):
        request = TextSearchRequest(
            query="test query",
            space_key="DEV",
            content_type="page",
            include_archived=False,
            limit=20,
            start=0,
        )
        json_data = request.model_dump_json(indent=2)
        assert '"query": "test query"' in json_data
        assert '"space_key": "DEV"' in json_data
        assert '"content_type": "page"' in json_data
        assert '"include_archived": false' in json_data
        assert '"limit": 20' in json_data
        assert '"start": 0' in json_data

        request = AdvancedSearchRequest(
            query="api documentation",
            sort_by=["updated_at", "title"],
            sort_direction=["desc", "asc"],
        )
        json_data = request.model_dump_json(indent=2)
        import json

        parsed_data = json.loads(json_data)
        assert parsed_data["query"] == "api documentation"
        assert parsed_data["sort_by"] == ["updated_at", "title"]
        assert parsed_data["sort_direction"] == ["desc", "asc"]

    def test_response_serialization(self):
        item = SearchResultItem(
            id="12345",
            title="API Documentation",
            type="page",
            space_key="DEV",
            space_name="Development",
            url="https://confluence.example.com/display/DEV/API+Documentation",
            last_modified=datetime.fromisoformat("2023-05-15T14:32:21"),
        )
        links = PaginationLinks(
            next="/api/search?query=api&start=20&limit=20", previous=None
        )
        response = SearchResponse(
            results=[item],
            total=42,
            start=0,
            limit=20,
            took_ms=123.45,
            page_count=3,
            current_page=1,
            has_more=True,
            links=links,
        )
        json_data = response.model_dump_json(indent=2)
        import json

        parsed_data = json.loads(json_data)
        assert "results" in parsed_data
        assert parsed_data["total"] == 42
        assert parsed_data["has_more"] is True
        assert parsed_data["links"]["next"] == "/api/search?query=api&start=20&limit=20"

        error = ErrorResponse(
            code=400,
            message="Invalid search parameters",
            details={
                "param": "query",
                "reason": "Query must be at least 2 characters long",
            },
        )
        json_data = error.model_dump_json(indent=2)
        assert '"status": "error"' in json_data
        assert '"code": 400' in json_data
        assert '"message": "Invalid search parameters"' in json_data
        assert '"details":' in json_data
        assert '"param": "query"' in json_data
