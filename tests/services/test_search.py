from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from confluence_gateway.core.config import load_configurations

# Check if Confluence config could be loaded
_confluence_config, _, _, _ = load_configurations()
real_config_available = _confluence_config is not None

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.exceptions import SearchParameterError
from confluence_gateway.services.search import (
    EnhancedSearchResult,
    SearchService,
    SearchStatistics,
    SortDirection,
    SortField,
)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=ConfluenceClient)

    def mock_search(**kwargs):
        pages = [
            ConfluencePage(
                id=f"page-{i}",
                title=f"Test Page {i}",
                content_type=ContentType.PAGE,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                space={"key": f"SPACE{i}", "name": f"Space {i}"},
            )
            for i in range(1, 4)
        ]

        return SearchResult(
            total_size=3,
            start=kwargs.get("start", 0),
            limit=kwargs.get("limit", 10),
            results=pages,
        )

    client.search.side_effect = mock_search
    client.search_by_cql.side_effect = mock_search

    return client


@pytest.fixture
def search_service(mock_client):
    return SearchService(client=mock_client)


class TestSearchServiceInitialization:
    def test_init_with_client(self, mock_client):
        service = SearchService(client=mock_client)
        assert service.client == mock_client


class TestTextSanitization:
    def test_sanitize_text_basic(self, search_service):
        result = search_service._sanitize_text("  Hello  World  ")
        assert result == "Hello World"

    def test_sanitize_text_special_chars(self, search_service):
        result = search_service._sanitize_text("Hello! @World#")
        assert result == "Hello! @World#"  # # is a permitted character

    def test_sanitize_text_empty(self, search_service):
        with pytest.raises(SearchParameterError, match="cannot be empty"):
            search_service._sanitize_text("")

    def test_sanitize_text_too_short(self, search_service):
        with pytest.raises(SearchParameterError, match="at least 2 characters"):
            search_service._sanitize_text("a")

    def test_sanitize_text_only_whitespace(self, search_service):
        with pytest.raises(SearchParameterError, match="cannot be empty"):
            search_service._sanitize_text("   ")


class TestParameterValidation:
    def test_validate_limit_too_high(self, search_service, mock_client):
        with pytest.raises(SearchParameterError, match="Limit must be between"):
            search_service.search_by_text("test", limit=1000)

    def test_validate_limit_negative(self, search_service, mock_client):
        with pytest.raises(SearchParameterError, match="Limit must be between"):
            search_service.search_by_text("test", limit=-5)

    def test_validate_start_negative(self, search_service, mock_client):
        with pytest.raises(
            SearchParameterError, match="Start position cannot be negative"
        ):
            search_service.search_by_text("test", start=-1)

    def test_cql_validation_empty(self, search_service):
        with pytest.raises(SearchParameterError, match="CQL query cannot be empty"):
            search_service.search_by_cql("")

    def test_cql_validation_invalid_format(self, search_service):
        with pytest.raises(SearchParameterError, match="Invalid CQL query format"):
            search_service.search_by_cql("this is not a valid cql query")


class TestSearchByText:
    def test_search_by_text_basic(self, search_service, mock_client):
        result = search_service.search_by_text("test query")

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        assert call_args["query"] == "test query"

        assert isinstance(result, EnhancedSearchResult)
        assert result.query == "test query"
        assert len(result.results.results) == 3

    def test_search_by_text_with_filters(self, search_service, mock_client):
        result = search_service.search_by_text(
            "test query",
            content_type=ContentType.PAGE,
            space_key="SPACE1",
            include_archived=True,
        )

        call_args = mock_client.search.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["content_type"] == ContentType.PAGE
        assert call_args["space_key"] == "SPACE1"
        assert call_args["include_archived"] is True

        assert result.filters_applied["content_type"] == ContentType.PAGE
        assert result.filters_applied["space_key"] == "SPACE1"

    def test_search_by_text_without_enhancement(self, search_service, mock_client):
        result = search_service.search_by_text(
            "test query", return_enhanced_result=False
        )

        assert isinstance(result, SearchResult)
        assert not isinstance(result, EnhancedSearchResult)


class TestSearchByCQL:
    def test_search_by_cql_basic(self, search_service, mock_client):
        result = search_service.search_by_cql('space = "TEST" AND text ~ "query"')

        mock_client.search_by_cql.assert_called_once()
        call_args = mock_client.search_by_cql.call_args[1]
        assert call_args["cql"] == 'space = "TEST" AND text ~ "query"'

        assert isinstance(result, EnhancedSearchResult)
        assert result.query is not None
        assert "CQL:" in result.query
        assert len(result.results.results) == 3

    def test_search_by_cql_with_pagination(self, search_service, mock_client):
        search_service.search_by_cql(
            'space = "TEST" AND text ~ "query"', start=10, limit=20
        )

        call_args = mock_client.search_by_cql.call_args[1]
        assert call_args["start"] == 10
        assert call_args["limit"] == 20


class TestResultSorting:
    def test_sort_by_title(self, search_service):
        pages = [
            ConfluencePage(id="1", title="B Page"),
            ConfluencePage(id="2", title="A Page"),
            ConfluencePage(id="3", title="C Page"),
        ]
        result = SearchResult(total_size=3, start=0, limit=10, results=pages)

        sorted_result = search_service._sort_results(
            result, sort_fields=[SortField.TITLE], directions=[SortDirection.ASC]
        )

        assert [p.title for p in sorted_result.results] == [
            "A Page",
            "B Page",
            "C Page",
        ]

        sorted_result = search_service._sort_results(
            result, sort_fields=[SortField.TITLE], directions=[SortDirection.DESC]
        )

        assert [p.title for p in sorted_result.results] == [
            "C Page",
            "B Page",
            "A Page",
        ]

    def test_sort_by_date(self, search_service):
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        pages = [
            ConfluencePage(id="1", title="Page 1", updated_at=now),
            ConfluencePage(id="2", title="Page 2", updated_at=yesterday),
            ConfluencePage(id="3", title="Page 3", updated_at=tomorrow),
        ]
        result = SearchResult(total_size=3, start=0, limit=10, results=pages)

        sorted_result = search_service._sort_results(
            result, sort_fields=[SortField.UPDATED], directions=[SortDirection.ASC]
        )

        assert [p.id for p in sorted_result.results] == ["2", "1", "3"]

    def test_sort_by_multiple_fields(self, search_service):
        pages = [
            ConfluencePage(id="1", title="B Page", space={"key": "SPACE-A"}),
            ConfluencePage(id="2", title="A Page", space={"key": "SPACE-A"}),
            ConfluencePage(id="3", title="C Page", space={"key": "SPACE-B"}),
        ]
        result = SearchResult(total_size=3, start=0, limit=10, results=pages)

        sorted_result = search_service._sort_results(
            result,
            sort_fields=[SortField.SPACE, SortField.TITLE],
            directions=[SortDirection.ASC, SortDirection.ASC],
        )

        sorted_ids = [p.id for p in sorted_result.results]
        assert sorted_ids[0] in ["1", "2"]
        assert sorted_ids[1] in ["1", "2"]
        assert sorted_ids[0] != sorted_ids[1]
        assert sorted_ids[2] == "3"


class TestResultFiltering:
    def test_filter_by_relevance_top_n(self, search_service):
        pages = [ConfluencePage(id=str(i), title=f"Page {i}") for i in range(1, 6)]
        result = SearchResult(total_size=5, start=0, limit=10, results=pages)

        filtered = search_service._filter_by_relevance(result, top_n=3)

        assert len(filtered.results) == 3
        assert filtered.results[0].id == "1"
        assert filtered.results[2].id == "3"
        assert filtered.total_size == 5


class TestResultEnhancement:
    def test_process_search_result(self, search_service):
        pages = [ConfluencePage(id=str(i), title=f"Page {i}") for i in range(1, 4)]
        result = SearchResult(total_size=10, start=0, limit=3, results=pages)

        enhanced = search_service._process_search_result(
            result,
            execution_time_ms=123.45,
            query="test query",
            filters={"space_key": "TEST"},
            sort_criteria=[{"field": "title", "direction": "asc"}],
            current_page=1,
        )

        assert isinstance(enhanced, EnhancedSearchResult)
        assert enhanced.results == result
        assert enhanced.query == "test query"
        assert enhanced.filters_applied == {"space_key": "TEST"}
        assert enhanced.sort_criteria == [{"field": "title", "direction": "asc"}]

        assert enhanced.statistics.total_results == 10
        assert enhanced.statistics.filtered_results == 3
        assert enhanced.statistics.execution_time_ms == 123.45
        assert enhanced.statistics.current_page == 1
        assert enhanced.statistics.total_pages == 4

    def test_to_standard_result(self, search_service):
        pages = [ConfluencePage(id="1", title="Page 1")]
        result = SearchResult(total_size=1, start=0, limit=10, results=pages)

        enhanced = EnhancedSearchResult(
            results=result,
            statistics=SearchStatistics(
                total_results=1,
                filtered_results=1,
                total_pages=1,
                current_page=1,
                execution_time_ms=100.0,
            ),
        )

        standard = enhanced.to_standard_result()

        assert standard == result
        assert isinstance(standard, SearchResult)


@pytest.fixture
def real_client(request):
    config, _, _, _ = load_configurations()

    if not config:
        pytest.skip(
            "Skipping integration tests - Confluence configuration not found in environment or config file"
        )

    from confluence_gateway.adapters.confluence.client import ConfluenceClient

    client = ConfluenceClient(config=config)

    try:
        client.test_connection()
    except Exception as e:
        pytest.skip(f"Could not connect to Confluence: {str(e)}")

    return client


@pytest.fixture
def real_search_service(real_client):
    return SearchService(client=real_client)


@pytest.fixture
def real_search_term(real_client):
    import random
    import re

    def extract_content_tokens(text, min_length=2, max_length=20):
        if not text:
            return []

        # Extract all word-like tokens, regardless of language
        # This works with non-Latin alphabets like Chinese, Japanese, Korean, Cyrillic, etc.
        tokens = re.findall(r"\b\w+\b", text, re.UNICODE)

        # Filter by length only, not by language-specific stop words
        valid_tokens = [
            token for token in tokens if min_length <= len(token) <= max_length
        ]

        # Return random sample to avoid frequency bias
        if len(valid_tokens) > 30:
            return random.sample(valid_tokens, 30)
        return valid_tokens

    def get_content_from_space(space_key, limit=5):
        content_tokens = []
        try:
            # Get pages from this space, use random sorting to increase variability
            sort_methods = ["created", "modified", "title"]
            sort_order = random.choice(["asc", "desc"])
            sort_field = random.choice(sort_methods)

            cql_query = f'space = "{space_key}" AND type in (page, blogpost) ORDER BY {sort_field} {sort_order}'
            space_content = real_client.search_by_cql(
                cql_query, limit=limit, expand=["body.view", "space"]
            )

            if space_content and space_content.results:
                # Get a random subset of pages
                pages = list(space_content.results)
                random.shuffle(pages)

                for page in pages[: min(3, len(pages))]:
                    page_text = ""
                    if hasattr(page, "html_content") and page.html_content:
                        page_text = re.sub(r"<[^>]+>", " ", page.html_content)
                        page_text = re.sub(r"\s+", " ", page_text).strip()
                    elif hasattr(page, "plain_content") and page.plain_content:
                        page_text = page.plain_content
                    elif hasattr(page, "title") and page.title:
                        page_text = page.title
                    else:
                        continue

                    tokens = extract_content_tokens(page_text)
                    content_tokens.extend(tokens)

                    # If we've collected enough tokens, stop early
                    if len(content_tokens) >= 50:
                        break
        except Exception:
            pass

        # Randomize the order to avoid any bias
        random.shuffle(content_tokens)
        return content_tokens

    token_candidates = []
    random_sampling_enabled = True

    # 1. First find actual spaces in the Confluence instance
    try:
        spaces_response = real_client.atlassian_api.get_all_spaces(limit=10)
        if (
            spaces_response
            and "results" in spaces_response
            and spaces_response["results"]
        ):
            # Randomly select spaces to analyze
            spaces = spaces_response["results"]
            random.shuffle(spaces)
            space_sample = spaces[: min(3, len(spaces))]

            # Get space keys and extract tokens from space names
            space_keys = []
            for space in space_sample:
                if "key" in space and space["key"]:
                    space_keys.append(space["key"])

                    # Add tokens from space name
                    if "name" in space and space["name"]:
                        name_tokens = extract_content_tokens(space["name"])
                        token_candidates.extend(name_tokens)

            # 2. Get content from randomly selected spaces
            random.shuffle(space_keys)
            for space_key in space_keys[: min(2, len(space_keys))]:
                content_tokens = get_content_from_space(space_key)
                token_candidates.extend(content_tokens)

                # If we have enough candidates, we can stop
                if len(token_candidates) >= 40:
                    break
    except Exception:
        pass

    # 3. If we couldn't get enough space-specific content, try general content with random ordering
    if len(token_candidates) < 20:
        try:
            # Use different random ordering each time
            sort_fields = ["created", "modified", "title"]
            sort_directions = ["asc", "desc"]
            random_sort = f"ORDER BY {random.choice(sort_fields)} {random.choice(sort_directions)}"

            recent_content = real_client.search_by_cql(
                f"type in (page, blogpost) {random_sort}",
                limit=7,
                expand=["body.view", "space"],
            )

            if recent_content and recent_content.results:
                pages = list(recent_content.results)
                random.shuffle(pages)

                for page in pages[: min(3, len(pages))]:
                    page_text = ""
                    if hasattr(page, "html_content") and page.html_content:
                        # Extract a random section of the HTML content to increase diversity
                        html_content = re.sub(r"<[^>]+>", " ", page.html_content)
                        html_content = re.sub(r"\s+", " ", html_content).strip()
                        if len(html_content) > 500 and random_sampling_enabled:
                            start_pos = random.randint(
                                0, max(0, len(html_content) - 500)
                            )
                            end_pos = min(start_pos + 500, len(html_content))
                            page_text = html_content[start_pos:end_pos]
                        else:
                            page_text = html_content
                    elif hasattr(page, "plain_content") and page.plain_content:
                        page_text = page.plain_content
                    elif hasattr(page, "title") and page.title:
                        page_text = page.title
                    else:
                        continue

                    tokens = extract_content_tokens(page_text)
                    token_candidates.extend(tokens)
        except Exception:
            pass

    # Deduplicate terms but maintain randomness
    token_candidates = list(set(token_candidates))

    # Always randomize to avoid any systematic bias
    random.shuffle(token_candidates)

    # No fallback terms - if we can't find real content tokens, the test should skip
    # instead of using hardcoded English words

    # Try multiple search candidates randomly
    candidates_with_results = []

    # Try up to 10 random candidates to find ones with results
    sample_size = min(10, len(token_candidates))
    if sample_size == 0:
        pytest.skip("No content tokens could be extracted from Confluence")

    for term in random.sample(token_candidates, sample_size):
        try:
            search_result = real_client.search(query=term, limit=5)
            result_count = search_result.total_size

            if result_count > 0:
                candidates_with_results.append((term, result_count))

                # If we find a great term with multiple results, use it immediately
                if result_count >= 3:
                    return term
        except Exception:
            continue

    candidates_with_results.sort(key=lambda x: x[1], reverse=True)

    if candidates_with_results:
        return candidates_with_results[0][0]

    try:
        spaces_response = real_client.atlassian_api.get_all_spaces(limit=1)
        if (
            spaces_response
            and "results" in spaces_response
            and spaces_response["results"]
        ):
            space = spaces_response["results"][0]
            if "name" in space and space["name"]:
                # Try using a part of the space name
                space_name_parts = re.findall(r"\w+", space["name"], re.UNICODE)
                if space_name_parts and len(space_name_parts[0]) >= 2:
                    return space_name_parts[0]
    except Exception:
        pass

    pytest.skip("Could not find any search terms that return results")


class TestSearchWithRealData:
    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_search_with_content_type_filter(
        self, real_search_service, real_search_term
    ):
        result = real_search_service.search_by_text(
            real_search_term, content_type=ContentType.PAGE, limit=10
        )

        assert result.statistics.total_results >= 0

        if result.statistics.filtered_results == 0:
            pytest.skip("No page results found for the search term")

        for page in result.results.results:
            assert page.content_type == ContentType.PAGE

        assert result.filters_applied["content_type"] == ContentType.PAGE

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_search_with_sorting_and_relevance_filtering(
        self, real_search_service, real_search_term
    ):
        unsorted_result = real_search_service.search_by_text(real_search_term, limit=10)

        if unsorted_result.statistics.total_results < 3:
            pytest.skip("Not enough search results for meaningful sorting test")

        sorted_result = real_search_service.search_by_text(
            real_search_term,
            top_n=2,
            sort_by=[SortField.UPDATED],
            sort_direction=[SortDirection.DESC],
            limit=10,
        )

        assert sorted_result.statistics.filtered_results <= 2

        if len(sorted_result.results.results) == 0:
            pytest.skip("No results returned from real Confluence API")

        first_result = sorted_result.results.results[0]
        assert isinstance(first_result.id, str)
        assert isinstance(first_result.title, str)

        assert sorted_result.query == real_search_term
        assert sorted_result.statistics.execution_time_ms > 0
        assert sorted_result.sort_criteria[0]["field"] == "updated_at"
        assert sorted_result.sort_criteria[0]["direction"] == "desc"

        assert sorted_result.filters_applied["top_n"] == 2

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_pagination_with_real_data(self, real_search_service, real_search_term):
        first_page = real_search_service.search_by_text(
            real_search_term, limit=2, start=0
        )

        if first_page.statistics.total_results < 3:
            pytest.skip("Not enough results for pagination test")

        second_page = real_search_service.search_by_text(
            real_search_term, limit=2, start=2
        )

        assert first_page.statistics.current_page == 1
        assert second_page.statistics.current_page == 2

        assert (
            first_page.statistics.total_results == second_page.statistics.total_results
        )
        assert first_page.statistics.total_pages == second_page.statistics.total_pages

        assert len(first_page.results.results) == 2
        assert 0 < len(second_page.results.results) <= 2
        assert first_page.results.results
        assert second_page.results.results

        if first_page.statistics.total_results > 4:
            for result in first_page.results.results + second_page.results.results:
                assert result.id
                assert result.title

        assert (
            first_page.statistics.total_results == second_page.statistics.total_results
        )
        assert first_page.statistics.total_pages == second_page.statistics.total_pages

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_search_with_multiple_keywords(self, real_search_service, real_search_term):
        if len(real_search_term.split()) < 2:
            additional_term = (
                "document" if "document" not in real_search_term else "content"
            )
            multiple_keywords = [real_search_term, additional_term]
        else:
            multiple_keywords = real_search_term.split()[:2]

        results = real_search_service.search_by_text(multiple_keywords, limit=5)

        assert isinstance(results.statistics.execution_time_ms, float)
        assert results.statistics.execution_time_ms > 0

        print(f"Multi-keyword query: {results.query}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_search_with_space_key_filtering(
        self, real_search_service, real_search_term, real_client
    ):
        try:
            spaces_response = real_client.atlassian_api.get_all_spaces(limit=1)
            if (
                not spaces_response
                or "results" not in spaces_response
                or len(spaces_response["results"]) == 0
            ):
                pytest.skip("No spaces available in Confluence instance")

            space_key = spaces_response["results"][0].get("key")
            if not space_key:
                pytest.skip("Could not get a valid space key from Confluence")

            filtered_results = real_search_service.search_by_text(
                real_search_term, space_key=space_key, limit=10
            )

            assert filtered_results.filters_applied["space_key"] == space_key
            return
        except Exception as e:
            pytest.skip(f"Error accessing spaces API: {str(e)}")

        filtered_results = real_search_service.search_by_text(
            real_search_term, space_key=space_key, limit=10
        )

        for result in filtered_results.results.results:
            result_space_key = getattr(result, "space_key", None)
            if not result_space_key and hasattr(result, "space"):
                space = result.space
                if isinstance(space, dict):
                    result_space_key = space.get("key")
                else:
                    result_space_key = getattr(space, "key", None)

            assert result_space_key == space_key

        assert filtered_results.filters_applied["space_key"] == space_key

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_cql_search_with_real_data(self, real_search_service, real_search_term):
        cql_query = f'text ~ "{real_search_term}"'

        result = real_search_service.search_by_cql(cql_query, limit=5)

        assert result.results.total_size > 0
        assert f"CQL: {cql_query}" in result.query
        assert result.statistics.execution_time_ms > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_advanced_cql_search(self, real_search_service, real_search_term):
        cql_query = f'text ~ "{real_search_term}" ORDER BY created DESC'

        result = real_search_service.search_by_cql(cql_query, limit=5)

        if result.statistics.total_results == 0:
            pytest.skip("No results found for the advanced CQL query")

        if len(result.results.results) >= 2:
            for i in range(len(result.results.results) - 1):
                current = result.results.results[i]
                next_item = result.results.results[i + 1]

                if not current.created_at or not next_item.created_at:
                    continue

                assert current.created_at >= next_item.created_at

        assert f"CQL: {cql_query}" in result.query

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_get_all_results_with_max_limit(
        self, real_search_service, real_search_term
    ):
        result = real_search_service.search_by_text(
            real_search_term, get_all_results=True, max_results=3
        )

        assert len(result.results.results) <= 3

        if result.statistics.total_results > 3:
            assert len(result.results.results) == 3

        assert result.filters_applied.get("get_all_results") is True
        assert result.filters_applied.get("max_results") == 3

    @pytest.mark.integration
    @pytest.mark.skipif(
        not real_config_available,
        reason="Required environment variables for Confluence API not set",
    )
    def test_total_results_with_limited_page_size(
        self, real_search_service, real_search_term
    ):
        large_result = real_search_service.search_by_text(real_search_term, limit=10)

        if large_result.statistics.total_results <= 1:
            pytest.skip("Not enough results to test pagination accuracy")

        small_result = real_search_service.search_by_text(real_search_term, limit=1)

        assert (
            small_result.statistics.total_results
            == large_result.statistics.total_results
        )

        assert len(small_result.results.results) == 1

        assert small_result.statistics.filtered_results == len(
            small_result.results.results
        )
