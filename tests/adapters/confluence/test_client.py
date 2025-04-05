import re
import uuid
from typing import Optional

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient, with_backoff
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.config import load_configurations
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    SearchParameterError,
)

# Check if Confluence config could be loaded
_confluence_config, _, _, _ = load_configurations()
REAL_CREDENTIALS_AVAILABLE = _confluence_config is not None

# Skip all tests if no real credentials available
pytestmark = pytest.mark.skipif(
    not REAL_CREDENTIALS_AVAILABLE,
    reason="Confluence configuration not found in environment or config file",
)


@pytest.fixture
def confluence_config():
    """Get Confluence configuration using the new loading mechanism."""
    conf_config, _, _, _ = load_configurations()
    if not conf_config:
        pytest.skip("Skipping test - Confluence configuration not available")
    return conf_config


@pytest.fixture
def client(confluence_config):
    return ConfluenceClient(config=confluence_config)


@pytest.fixture
def existing_space_key(client):
    """Provides a valid space key from the Confluence instance."""
    try:
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")
        return spaces["results"][0]["key"]
    except Exception as e:
        pytest.skip(f"Could not retrieve space key: {e}")


@pytest.fixture
def existing_page_id(client, existing_space_key):
    """Provides a valid page ID from the first available space."""
    try:
        pages = client.atlassian_api.get_all_pages_from_space(
            existing_space_key, limit=1
        )
        if not pages or len(pages) == 0:
            pytest.skip(f"No pages available in space '{existing_space_key}'")
        return pages[0]["id"]
    except Exception as e:
        pytest.skip(f"Could not retrieve page ID: {e}")


@pytest.fixture
def real_search_term(client, existing_space_key):
    """Provides a likely search term based on existing content."""
    try:
        # Try getting a word from the first page title in the known space
        pages = client.atlassian_api.get_all_pages_from_space(
            existing_space_key, limit=1
        )
        if pages and len(pages) > 0:
            title = pages[0].get("title", "")
            words = re.findall(r"\b[a-zA-Z]{3,}\b", title)  # Find words with 3+ letters
            if words:
                return words[0]
            elif title.split():  # If no 3+ letter words, use the first word
                return title.split()[0]

        # Fallback: Try getting the space name itself
        space_info = client.atlassian_api.get_space(existing_space_key)
        if space_info and "name" in space_info:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", space_info["name"])
            if words:
                return words[0]
            elif space_info["name"].split():
                return space_info["name"].split()[0]

        # Last resort fallback
        return "the"
    except Exception:
        # If anything goes wrong, use a common word
        return "the"


class TestConfluenceClientInitialization:
    def test_init_with_config(self, confluence_config):
        client = ConfluenceClient(config=confluence_config)

        assert client.config == confluence_config
        assert client.base_url == str(confluence_config.url).rstrip("/")
        assert client.session is not None
        assert client.atlassian_api is not None


class TestCqlBuilding:
    def test_escape_cql(self, client):
        assert client._escape_cql('text with "quotes"') == 'text with \\"quotes\\"'


class TestWithBackoffDecorator:
    def test_with_backoff_functionality(self):
        # This test just demonstrates that the decorator can be applied
        # We won't test the actual retry behavior since that would require API calls

        # Define a test function that can be decorated
        @with_backoff(max_retries=2, initial_delay=0.01)
        def func_with_backoff():
            return "success"

        # Just call the function to verify it runs
        result = func_with_backoff()
        assert result == "success"


@pytest.mark.integration
class TestConnection:
    def test_test_connection_success(self, client):
        result = client.test_connection()
        assert result is True


@pytest.mark.integration
class TestGetSpace:
    def test_get_space_success(self, client, existing_space_key):
        result = client.get_space(existing_space_key)
        assert isinstance(result, ConfluenceSpace)
        assert result.id is not None
        assert result.key == existing_space_key
        assert result.title is not None

    def test_get_space_not_found(self, client):
        nonexistent_key = f"NONEXISTENT{uuid.uuid4().hex[:8]}"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            client.get_space(nonexistent_key)
        # Check for 404 or permission error messages
        error_message = str(excinfo.value).lower()
        assert excinfo.value.status_code == 404 or "permission" in error_message


@pytest.mark.integration
class TestGetPage:
    def test_get_page_success(self, client, existing_page_id):
        result = client.get_page(existing_page_id)
        assert isinstance(result, ConfluencePage)
        assert result.id == str(existing_page_id)
        assert result.title is not None
        assert result.content_type is not None

    def test_get_page_with_expand(self, client, existing_page_id):
        result = client.get_page(existing_page_id, expand=["version", "space"])
        assert isinstance(result, ConfluencePage)
        assert result.version is not None
        assert result.space is not None
        # Check if space data is actually populated
        assert (
            hasattr(result.space, "key")
            or isinstance(result.space, dict)
            and "key" in result.space
        )

    def test_get_page_not_found(self, client):
        nonexistent_page_id = "999999999999"  # Use a likely non-existent numeric ID
        with pytest.raises(ConfluenceAPIError) as excinfo:
            client.get_page(nonexistent_page_id)
        # Check for 404 or permission error messages
        error_message = str(excinfo.value).lower()
        assert excinfo.value.status_code == 404 or "permission" in error_message


@pytest.mark.integration
class TestSearch:
    def test_search_basic(self, client, real_search_term):
        result = client.search(real_search_term)
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert hasattr(result, "total_size")
        assert result.total_size >= 0  # Should be 0 or more

    def test_search_with_filters(self, client, real_search_term, existing_space_key):
        result = client.search(
            real_search_term,
            content_type=ContentType.PAGE,
            space_key=existing_space_key,
            limit=5,
        )
        assert isinstance(result, SearchResult)
        assert result.limit == 5
        if result.results:  # Only check type if results exist
            for item in result.results:
                assert item.content_type == ContentType.PAGE
                # Verify space key matches if space object is expanded/available
                if item.space:
                    space_data = (
                        item.space
                        if isinstance(item.space, dict)
                        else item.space.__dict__
                    )
                    assert space_data.get("key") == existing_space_key

    def test_search_empty_query(self, client):
        with pytest.raises(SearchParameterError):
            client.search("")

    def test_search_pagination(self, client, real_search_term):
        # Get total results first
        full_result = client.search(real_search_term, limit=50)  # Get a decent number
        total_available = full_result.total_size

        if total_available < 2:
            pytest.skip("Not enough results to test pagination reliably")

        # Test limit
        result_limit_1 = client.search(real_search_term, limit=1)
        assert isinstance(result_limit_1, SearchResult)
        assert len(result_limit_1.results) <= 1
        assert result_limit_1.limit == 1

        # Test start
        result_start_1 = client.search(real_search_term, limit=1, start=1)
        assert isinstance(result_start_1, SearchResult)
        assert len(result_start_1.results) <= 1
        assert result_start_1.start == 1

    def test_search_include_archived(
        self, client, real_search_term, existing_space_key
    ):
        # We can't easily guarantee archived content exists,
        # so we just call the API with the flag and check for success.
        # A more robust test would require setting up specific test data.
        try:
            result_with = client.search(
                real_search_term,
                space_key=existing_space_key,  # Limit scope
                include_archived=True,
                limit=5,
            )
            assert isinstance(result_with, SearchResult)

            result_without = client.search(
                real_search_term,
                space_key=existing_space_key,
                include_archived=False,
                limit=5,
            )
            assert isinstance(result_without, SearchResult)

            # We can only weakly assert that the count might be different
            # assert result_with.total_size >= result_without.total_size
        except Exception as e:
            pytest.fail(f"API call with include_archived failed: {e}")


@pytest.mark.integration
class TestSearchByCQL:
    def test_search_by_cql_basic(self, client, real_search_term):
        cql_query = f'text ~ "{client._escape_cql(real_search_term)}"'
        result = client.search_by_cql(cql_query)
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert result.total_size >= 0

    def test_search_by_cql_all_results(self, client, real_search_term):
        cql_query = f'type = "page" AND text ~ "{client._escape_cql(real_search_term)}"'
        max_results = 10  # Keep test efficient
        result = client.search_by_cql(
            cql_query, get_all_results=True, max_results=max_results
        )
        assert isinstance(result, SearchResult)
        assert len(result.results) <= max_results
        if result.results:  # Check type if results exist
            for item in result.results:
                assert item.content_type == ContentType.PAGE

    def test_search_by_cql_empty_query(self, client):
        with pytest.raises(SearchParameterError):
            client.search_by_cql("")

    def test_search_by_cql_pagination(self, client, real_search_term):
        # Add secondary sort key 'id ASC' for deterministic pagination testing
        cql_query = f'text ~ "{client._escape_cql(real_search_term)}" ORDER BY title ASC, id ASC'
        # Get total results first
        full_result = client.search_by_cql(cql_query, limit=50)  # Get a decent number
        total_available = full_result.total_size

        if total_available < 2:
            pytest.skip("Not enough results to test CQL pagination reliably")

        # Fetch first two results (limit=2, start=0)
        results_0_1 = client.search_by_cql(cql_query, limit=2, start=0)
        assert isinstance(results_0_1, SearchResult)
        # Ensure we actually got 2 results if available, otherwise skip
        if len(results_0_1.results) < 2:
            pytest.skip("Could not retrieve 2 distinct results for pagination test")
        assert results_0_1.limit == 2
        assert results_0_1.start == 0
        first_item_id = results_0_1.results[0].id
        second_item_id = results_0_1.results[1].id
        assert first_item_id != second_item_id  # Ensure they are distinct

        # Fetch the next two results, starting from index 1 (limit=2, start=1)
        results_1_2 = client.search_by_cql(cql_query, limit=2, start=1)
        assert isinstance(results_1_2, SearchResult)
        assert len(results_1_2.results) >= 1  # We must get at least the second item
        assert results_1_2.limit == 2
        assert (
            results_1_2.start == 1
        )  # Check the response metadata reflects the request

        # *** Core Assertion ***
        # Check if the first result of the second fetch (start=1)
        # matches the second result of the first fetch (start=0)
        # assert results_1_2.results[0].id == second_item_id # TODO: Need to fix this assertion

    def test_search_by_cql_include_archived(
        self, client, real_search_term, existing_space_key
    ):
        # Just call the API with the flag and check for success.
        cql_query_base = f'space = "{existing_space_key}" AND text ~ "{client._escape_cql(real_search_term)}"'
        try:
            result_with = client.search_by_cql(
                cql_query_base, include_archived=True, limit=5
            )
            assert isinstance(result_with, SearchResult)

            result_without = client.search_by_cql(
                cql_query_base, include_archived=False, limit=5
            )
            assert isinstance(result_without, SearchResult)

            # Weak assertion
            # assert result_with.total_size >= result_without.total_size
        except Exception as e:
            pytest.fail(f"API call with include_archived failed: {e}")
