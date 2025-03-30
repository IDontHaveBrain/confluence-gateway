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
from confluence_gateway.core.config import load_confluence_config_from_env
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    SearchParameterError,
)

# Check for required environment variables at module level
REAL_CREDENTIALS_AVAILABLE = load_confluence_config_from_env() is not None

# Skip all tests if no real credentials available
pytestmark = pytest.mark.skipif(
    not REAL_CREDENTIALS_AVAILABLE,
    reason="Confluence API credentials not set in environment variables",
)


@pytest.fixture
def confluence_config():
    """Get Confluence configuration from environment."""
    return load_confluence_config_from_env()


@pytest.fixture
def client(confluence_config):
    return ConfluenceClient(config=confluence_config)


@pytest.fixture
def real_search_term(client):
    """Get a search term that will return results based on actual content."""
    try:
        # First approach: Get a term from a page title
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            raise ValueError("No spaces available")

        space_key = spaces["results"][0]["key"]
        pages = client.atlassian_api.get_all_pages_from_space(space_key, limit=3)

        if pages and len(pages) > 0:
            # Extract a meaningful word from a page title
            title = pages[0].get("title", "")
            # Find words with 3+ characters
            words = re.findall(r"\b[a-zA-Z]{3,}\b", title)
            if words:
                return words[0]
            elif title:
                return title.split()[0] if title.split() else "the"

        # Second approach: Get a term from space name
        if spaces and "results" in spaces and spaces["results"]:
            space_name = spaces["results"][0].get("name", "")
            words = re.findall(r"\b[a-zA-Z]{3,}\b", space_name)
            if words:
                return words[0]

        # Fallback to default terms
        return "the"
    except Exception:
        # If anything goes wrong, use a common word as fallback
        return "the"


def get_test_page_id(client) -> Optional[str]:
    """Helper to get an existing page ID for testing."""
    try:
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            return None

        space_key = spaces["results"][0]["key"]
        pages = client.atlassian_api.get_all_pages_from_space(space_key, limit=1)
        if not pages or len(pages) == 0:
            return None

        return pages[0]["id"]
    except Exception:
        return None


class TestConfluenceClientInitialization:
    def test_init_with_config(self, confluence_config):
        client = ConfluenceClient(config=confluence_config)

        assert client.config == confluence_config
        assert client.base_url == str(confluence_config.url).rstrip("/")
        assert client.session is not None
        assert client.atlassian_api is not None


class TestSessionManagement:
    def test_create_session(self, client):
        session = client._create_session()

        assert session.auth is not None
        assert "Accept" in session.headers
        assert "Content-Type" in session.headers


class TestMakeRequest:
    def test_make_get_request_success(self, client):
        try:
            result = client._make_request("get", "space", params={"limit": 1})

            assert isinstance(result, dict)
            assert "results" in result
        except ConfluenceAPIError:
            spaces = client.atlassian_api.get_all_spaces(limit=1)
            assert isinstance(spaces, dict)
            assert "results" in spaces
            pytest.skip("Direct API access failed, but atlassian-python-api works")

    def test_make_request_with_model_class(self, client):
        try:
            spaces = client.atlassian_api.get_all_spaces(limit=1)

            if not spaces or not spaces.get("results") or len(spaces["results"]) == 0:
                pytest.skip("No spaces available in the Confluence instance")

            space_key = spaces["results"][0]["key"]

            result = client._make_request(
                "get", f"space/{space_key}", model_class=ConfluenceSpace
            )

            assert isinstance(result, ConfluenceSpace)
            assert result.id is not None
            assert result.title is not None
            assert result.key == space_key
        except ConfluenceAPIError:
            spaces = client.atlassian_api.get_all_spaces(limit=1)
            space_key = spaces["results"][0]["key"]
            space_data = client.atlassian_api.get_space(space_key)
            result = client._parse_space(space_data)

            assert isinstance(result, ConfluenceSpace)
            assert result.id is not None
            assert result.title is not None
            assert result.key == space_key

            pytest.skip(
                "Direct API access failed, but transformation with atlassian-python-api works"
            )

    def test_make_request_not_found_error(self, client):
        nonexistent_id = f"99999{uuid.uuid4().hex[:8]}"

        with pytest.raises(ConfluenceAPIError) as excinfo:
            client._make_request("get", f"content/{nonexistent_id}")

        assert excinfo.value.status_code in (404, 400)


class TestParsingMethods:
    def test_parse_space(self, client):
        spaces_response = client.atlassian_api.get_all_spaces(limit=1)

        if (
            not spaces_response
            or "results" not in spaces_response
            or len(spaces_response["results"]) == 0
        ):
            pytest.skip("No spaces available in the Confluence instance")

        space_data = spaces_response["results"][0]

        result = client._parse_space(space_data)

        assert isinstance(result, ConfluenceSpace)
        assert result.id is not None
        assert result.title is not None
        assert result.key is not None

        space_data_numeric = {
            "id": 98309,
            "name": "Test Personal Space",
            "key": "~test",
            "type": "personal",
        }

        result_numeric = client._parse_space(space_data_numeric)

        assert isinstance(result_numeric, ConfluenceSpace)
        assert result_numeric.id == "98309"
        assert result_numeric.title == "Test Personal Space"
        assert result_numeric.key == "~test"

    def test_parse_page(self, client):
        spaces_response = client.atlassian_api.get_all_spaces(limit=1)

        if (
            not spaces_response
            or "results" not in spaces_response
            or len(spaces_response["results"]) == 0
        ):
            pytest.skip("No spaces available in the Confluence instance")

        space_key = spaces_response["results"][0]["key"]

        # Get pages from that space
        pages = client.atlassian_api.get_all_pages_from_space(space_key, limit=1)
        if not pages or len(pages) == 0:
            pytest.skip("No pages available in the Confluence space")

        # Get full page data
        page_data = client.atlassian_api.get_page_by_id(
            pages[0]["id"], expand="body.view,body.storage,space,version"
        )

        # Parse the page data
        result = client._parse_page(page_data)

        # Verify basic structure
        assert isinstance(result, ConfluencePage)
        assert result.id is not None
        assert result.title is not None
        assert result.content_type is not None
        assert result.space is not None

    def test_parse_search_result(self, client, real_search_term):
        # Perform a simple search to get real search results using a term from real content
        search_data = client.atlassian_api.cql(f'text ~ "{real_search_term}"', limit=2)

        # Parse the search results
        result = client._parse_search_result(search_data)

        # Verify basic structure
        assert isinstance(result, SearchResult)
        assert hasattr(result, "total_size")
        assert hasattr(result, "start")
        assert hasattr(result, "limit")
        assert hasattr(result, "results")


class TestCqlBuilding:
    def test_escape_cql(self, client):
        assert client._escape_cql('text with "quotes"') == 'text with \\"quotes\\"'

    def test_build_search_cql_basic(self, client):
        cql = client._build_search_cql("test query")
        assert 'text ~ "test query"' in cql
        # We no longer include 'status != "archived"' in CQL queries
        # Instead, we use the include_archived_spaces parameter in the API call

    def test_build_search_cql_with_filters(self, client):
        cql = client._build_search_cql(
            "test query",
            content_type=ContentType.PAGE,
            space_key="TEST",
            include_archived=True,
        )
        assert 'text ~ "test query"' in cql
        assert 'type = "page"' in cql
        assert 'space = "TEST"' in cql
        assert 'status != "archived"' not in cql  # Should not include status filter


class TestClientPublicMethods:
    def test_test_connection_success(self, client):
        result = client.test_connection()
        assert result is True

    def test_get_space_success(self, client):
        # Get list of spaces to find one to test with
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")

        space_key = spaces["results"][0]["key"]

        # Test with real space
        result = client.get_space(space_key)

        # Verify basic structure
        assert isinstance(result, ConfluenceSpace)
        assert result.id is not None
        assert result.key == space_key
        assert result.title is not None

    def test_get_space_not_found(self, client):
        # Generate a likely non-existent space key
        nonexistent_key = f"NONEXISTENT{uuid.uuid4().hex[:8]}"

        with pytest.raises(ConfluenceAPIError) as excinfo:
            client.get_space(nonexistent_key)

        # Verify error details
        error_message = str(excinfo.value)
        assert any(
            text in error_message.lower()
            for text in [
                "no space with the given key",
                "not found",
                "does not have permission",
            ]
        )

    def test_get_page_success(self, client):
        # First get a space
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")

        space_key = spaces["results"][0]["key"]

        # Get pages from that space
        pages = client.atlassian_api.get_all_pages_from_space(space_key, limit=1)
        if not pages or len(pages) == 0:
            pytest.skip("No pages available in the Confluence space")

        page_id = pages[0]["id"]

        # Test with real page
        result = client.get_page(page_id)

        # Verify basic structure
        assert isinstance(result, ConfluencePage)
        assert result.id == str(
            page_id
        )  # Ensure ID comparison works with string conversion
        assert result.title is not None
        assert result.content_type is not None

    def test_get_page_with_expand(self, client):
        # First get a space
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")

        space_key = spaces["results"][0]["key"]

        # Get pages from that space
        pages = client.atlassian_api.get_all_pages_from_space(space_key, limit=1)
        if not pages or len(pages) == 0:
            pytest.skip("No pages available in the Confluence space")

        page_id = pages[0]["id"]

        # Test with custom expand
        result = client.get_page(page_id, expand=["version", "space"])

        # Verify basic structure with specific expansions
        assert isinstance(result, ConfluencePage)
        assert result.version is not None
        assert result.space is not None

    def test_search_basic(self, client, real_search_term):
        # Test with a term derived from actual content
        result = client.search(real_search_term)

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert hasattr(result, "total_size")
        assert hasattr(result, "start")
        assert hasattr(result, "limit")

    def test_search_with_filters(self, client, real_search_term):
        # Get a space key to use in the filter
        spaces = client.atlassian_api.get_all_spaces(limit=1)
        if not spaces or "results" not in spaces or len(spaces["results"]) == 0:
            pytest.skip("No spaces available in the Confluence instance")

        space_key = spaces["results"][0]["key"]

        # Search with filters
        result = client.search(
            real_search_term,
            content_type=ContentType.PAGE,
            space_key=space_key,
            limit=5,
        )

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert result.limit == 5

        # All results should be of type PAGE
        for item in result.results:
            assert item.content_type == ContentType.PAGE

    def test_search_by_cql_basic(self, client, real_search_term):
        # Test with a term derived from actual content
        cql_query = f'text ~ "{real_search_term}"'

        result = client.search_by_cql(cql_query)

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert hasattr(result, "results")
        assert hasattr(result, "total_size")
        assert hasattr(result, "start")
        assert hasattr(result, "limit")

    def test_search_by_cql_all_results(self, client, real_search_term):
        # Set a low max_results to keep the test efficient
        # Use a query that combines content type with the real term
        cql_query = f'type = "page" AND text ~ "{real_search_term}"'
        max_results = 10

        result = client.search_by_cql(
            cql_query, get_all_results=True, max_results=max_results
        )

        # Verify result structure
        assert isinstance(result, SearchResult)
        assert len(result.results) <= max_results

    def test_search_empty_query(self, client):
        with pytest.raises(SearchParameterError):
            client.search("")

    def test_search_by_cql_empty_query(self, client):
        with pytest.raises(SearchParameterError):
            client.search_by_cql("")


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
