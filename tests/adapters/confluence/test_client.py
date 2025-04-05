import uuid

import pytest
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    SearchParameterError,
)

from tests.conftest import REAL_CONFIG_SKIP_REASON


@pytest.fixture
def existing_space_key(confluence_client):
    try:
        spaces = confluence_client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")
        return spaces["results"][0]["key"]
    except Exception as e:
        pytest.skip(f"Could not retrieve space key: {e}")


@pytest.fixture
def existing_page_id(confluence_client, existing_space_key):
    try:
        pages = confluence_client.atlassian_api.get_all_pages_from_space(
            existing_space_key, limit=1
        )
        if not pages or len(pages) == 0:
            pytest.skip(f"No pages available in space '{existing_space_key}'")
        return pages[0]["id"]
    except Exception as e:
        pytest.skip(f"Could not retrieve page ID: {e}")


class TestCqlBuilding:
    def test_escape_cql(self, confluence_client):
        assert (
            confluence_client._escape_cql('text with "quotes"')
            == 'text with \\"quotes\\"'
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestConnection:
    def test_test_connection_success(self, confluence_client):
        result = confluence_client.test_connection()
        assert result is True


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestGetSpace:
    def test_get_space_success(self, confluence_client, existing_space_key):
        result = confluence_client.get_space(existing_space_key)
        assert isinstance(result, ConfluenceSpace)
        assert result.id is not None
        assert result.key == existing_space_key
        assert result.title is not None

    def test_get_space_not_found(self, confluence_client):
        nonexistent_key = f"NONEXISTENT{uuid.uuid4().hex[:8]}"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.get_space(nonexistent_key)
        error_message = str(excinfo.value).lower()
        assert excinfo.value.status_code == 404 or "permission" in error_message


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestGetPage:
    def test_get_page_success(self, confluence_client, existing_page_id):
        result = confluence_client.get_page(existing_page_id)
        assert isinstance(result, ConfluencePage)
        assert result.id == str(existing_page_id)
        assert result.title is not None
        assert result.content_type is not None

    def test_get_page_with_expand(self, confluence_client, existing_page_id):
        result = confluence_client.get_page(
            existing_page_id, expand=["version", "space"]
        )
        assert isinstance(result, ConfluencePage)
        assert result.version is not None
        assert result.space is not None
        assert (
            hasattr(result.space, "key")
            or isinstance(result.space, dict)
            and "key" in result.space
        )

    def test_get_page_not_found(self, confluence_client):
        nonexistent_page_id = "999999999999"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.get_page(nonexistent_page_id)
        error_message = str(excinfo.value).lower()
        assert excinfo.value.status_code == 404 or "permission" in error_message


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestSearch:
    def test_search_basic(self, confluence_client, real_search_term):
        result = confluence_client.search(real_search_term)
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert hasattr(result, "total_size")
        assert result.total_size >= 0

    def test_search_with_filters(
        self, confluence_client, real_search_term, existing_space_key
    ):
        result = confluence_client.search(
            real_search_term,
            content_type=ContentType.PAGE,
            space_key=existing_space_key,
            limit=5,
        )
        assert isinstance(result, SearchResult)
        assert result.limit == 5
        if result.results:
            for item in result.results:
                assert item.content_type == ContentType.PAGE
                if item.space:
                    space_data = (
                        item.space
                        if isinstance(item.space, dict)
                        else item.space.__dict__
                    )
                    assert space_data.get("key") == existing_space_key

    def test_search_empty_query(self, confluence_client):
        with pytest.raises(SearchParameterError):
            confluence_client.search("")

    def test_search_pagination(self, confluence_client, real_search_term):
        full_result = confluence_client.search(real_search_term, limit=50)
        total_available = full_result.total_size

        if total_available < 2:
            pytest.skip("Not enough results to test pagination reliably")

        result_limit_1 = confluence_client.search(real_search_term, limit=1)
        assert isinstance(result_limit_1, SearchResult)
        assert len(result_limit_1.results) <= 1
        assert result_limit_1.limit == 1

        result_start_1 = confluence_client.search(real_search_term, limit=1, start=1)
        assert isinstance(result_start_1, SearchResult)
        assert len(result_start_1.results) <= 1
        assert result_start_1.start == 1

    def test_search_include_archived(
        self, confluence_client, real_search_term, existing_space_key
    ):
        try:
            result_with = confluence_client.search(
                real_search_term,
                space_key=existing_space_key,
                include_archived=True,
                limit=5,
            )
            assert isinstance(result_with, SearchResult)

            result_without = confluence_client.search(
                real_search_term,
                space_key=existing_space_key,
                include_archived=False,
                limit=5,
            )
            assert isinstance(result_without, SearchResult)

        except Exception as e:
            pytest.fail(f"API call with include_archived failed: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestSearchByCQL:
    def test_search_by_cql_basic(self, confluence_client, real_search_term):
        cql_query = f'text ~ "{confluence_client._escape_cql(real_search_term)}"'
        result = confluence_client.search_by_cql(cql_query)
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert result.total_size >= 0

    def test_search_by_cql_all_results(self, confluence_client, real_search_term):
        cql_query = f'type = "page" AND text ~ "{confluence_client._escape_cql(real_search_term)}"'
        max_results = 10
        result = confluence_client.search_by_cql(
            cql_query, get_all_results=True, max_results=max_results
        )
        assert isinstance(result, SearchResult)
        assert len(result.results) <= max_results
        if result.results:
            for item in result.results:
                assert item.content_type == ContentType.PAGE

    def test_search_by_cql_empty_query(self, confluence_client):
        with pytest.raises(SearchParameterError):
            confluence_client.search_by_cql("")

    def test_search_by_cql_pagination(self, confluence_client, real_search_term):
        cql_query = f'text ~ "{confluence_client._escape_cql(real_search_term)}" ORDER BY title ASC, id ASC'
        full_result = confluence_client.search_by_cql(cql_query, limit=50)
        total_available = full_result.total_size

        if total_available < 2:
            pytest.skip("Not enough results to test CQL pagination reliably")

        results_0_1 = confluence_client.search_by_cql(cql_query, limit=2, start=0)
        assert isinstance(results_0_1, SearchResult)
        if len(results_0_1.results) < 2:
            pytest.skip("Could not retrieve 2 distinct results for pagination test")
        assert results_0_1.limit == 2
        assert results_0_1.start == 0
        first_item_id = results_0_1.results[0].id
        second_item_id = results_0_1.results[1].id
        assert first_item_id != second_item_id

        results_1_2 = confluence_client.search_by_cql(cql_query, limit=2, start=1)
        assert isinstance(results_1_2, SearchResult)
        assert len(results_1_2.results) >= 1
        assert results_1_2.limit == 2
        assert results_1_2.start == 1

    def test_search_by_cql_include_archived(
        self, confluence_client, real_search_term, existing_space_key
    ):
        cql_query_base = f'space = "{existing_space_key}" AND text ~ "{confluence_client._escape_cql(real_search_term)}"'
        try:
            result_with = confluence_client.search_by_cql(
                cql_query_base, include_archived=True, limit=5
            )
            assert isinstance(result_with, SearchResult)

            result_without = confluence_client.search_by_cql(
                cql_query_base, include_archived=False, limit=5
            )
            assert isinstance(result_without, SearchResult)

        except Exception as e:
            pytest.fail(f"API call with include_archived failed: {e}")
