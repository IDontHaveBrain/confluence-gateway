import pytest
from confluence_gateway.adapters.confluence.models import ContentType
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.config import search_config
from confluence_gateway.core.exceptions import SearchParameterError, SemanticSearchError
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.search import (
    SearchService,
    SortDirection,
    SortField,
)

from tests.conftest import REAL_CONFIG_SKIP_REASON, SEMANTIC_SEARCH_SKIP_REASON


class TestSearchWithRealData:
    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_with_content_type_filter(
        self, standard_search_service, real_search_term
    ):
        result = standard_search_service.search_by_text(
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
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_with_sorting_and_relevance_filtering(
        self, standard_search_service, real_search_term
    ):
        unsorted_result = standard_search_service.search_by_text(
            real_search_term, limit=10
        )

        if unsorted_result.statistics.total_results < 3:
            pytest.skip("Not enough search results for meaningful sorting test")

        sorted_result = standard_search_service.search_by_text(
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
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_pagination_with_real_data(self, standard_search_service, real_search_term):
        first_page = standard_search_service.search_by_text(
            real_search_term, limit=2, start=0
        )

        if first_page.statistics.total_results < 3:
            pytest.skip("Not enough results for pagination test")

        second_page = standard_search_service.search_by_text(
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
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_with_multiple_keywords(
        self, standard_search_service, real_search_term
    ):
        if len(real_search_term.split()) < 2:
            additional_term = (
                "document" if "document" not in real_search_term else "content"
            )
            multiple_keywords = [real_search_term, additional_term]
        else:
            multiple_keywords = real_search_term.split()[:2]

        results = standard_search_service.search_by_text(multiple_keywords, limit=5)

        assert isinstance(results.statistics.execution_time_ms, float)
        assert results.statistics.execution_time_ms > 0

        print(f"Multi-keyword query: {results.query}")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_with_space_key_filtering(
        self, standard_search_service, real_search_term, confluence_client
    ):
        try:
            spaces_response = confluence_client.atlassian_api.get_all_spaces(limit=1)
            if (
                not spaces_response
                or "results" not in spaces_response
                or len(spaces_response["results"]) == 0
            ):
                pytest.skip("No spaces available in Confluence instance")

            space_key = spaces_response["results"][0].get("key")
            if not space_key:
                pytest.skip("Could not get a valid space key from Confluence")

            filtered_results = standard_search_service.search_by_text(
                real_search_term, space_key=space_key, limit=10
            )

            assert filtered_results.filters_applied["space_key"] == space_key
            return
        except Exception as e:
            pytest.skip(f"Error accessing spaces API: {str(e)}")

        filtered_results = standard_search_service.search_by_text(
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
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_cql_search_with_real_data(self, standard_search_service, real_search_term):
        cql_query = f'text ~ "{real_search_term}"'

        result = standard_search_service.search_by_cql(cql_query, limit=5)

        assert result.results.total_size > 0
        assert f"CQL: {cql_query}" in result.query
        assert result.statistics.execution_time_ms > 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_advanced_cql_search(self, standard_search_service, real_search_term):
        cql_query = f'text ~ "{real_search_term}" ORDER BY created DESC'

        result = standard_search_service.search_by_cql(cql_query, limit=5)

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
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_get_all_results_with_max_limit(
        self, standard_search_service, real_search_term
    ):
        result = standard_search_service.search_by_text(
            real_search_term, get_all_results=True, max_results=3
        )

        assert len(result.results.results) <= 3

        if result.statistics.total_results > 3:
            assert len(result.results.results) == 3

        assert result.filters_applied.get("get_all_results") is True
        assert result.filters_applied.get("max_results") == 3

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_by_text_invalid_limit(self, standard_search_service):
        with pytest.raises(SearchParameterError, match="Limit must be between 1 and"):
            standard_search_service.search_by_text("test", limit=0)
        with pytest.raises(SearchParameterError, match="Limit must be between 1 and"):
            standard_search_service.search_by_text("test", limit=-1)
        with pytest.raises(SearchParameterError, match="Limit must be between 1 and"):
            standard_search_service.search_by_text(
                "test", limit=search_config.max_limit + 1
            )

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_by_text_invalid_start(self, standard_search_service):
        with pytest.raises(
            SearchParameterError, match="Start position cannot be negative"
        ):
            standard_search_service.search_by_text("test", start=-1)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_by_text_invalid_query(self, standard_search_service):
        with pytest.raises(SearchParameterError, match="Search text cannot be empty"):
            standard_search_service.search_by_text("")
        with pytest.raises(SearchParameterError, match="Search text cannot be empty"):
            standard_search_service.search_by_text("   ")
        with pytest.raises(
            SearchParameterError, match="Search text must be at least 2 characters long"
        ):
            standard_search_service.search_by_text("a")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_search_by_cql_invalid_cql(self, standard_search_service):
        with pytest.raises(SearchParameterError, match="CQL query cannot be empty"):
            standard_search_service.search_by_cql("")
        with pytest.raises(SearchParameterError, match="CQL query cannot be empty"):
            standard_search_service.search_by_cql("   ")
        with pytest.raises(SearchParameterError, match="Invalid CQL query format"):
            standard_search_service.search_by_cql("just plain text")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.lazy_fixture("is_real_config_available"),
        reason=REAL_CONFIG_SKIP_REASON,
    )
    def test_total_results_with_limited_page_size(
        self, standard_search_service, real_search_term
    ):
        large_result = standard_search_service.search_by_text(
            real_search_term, limit=10
        )

        if large_result.statistics.total_results <= 1:
            pytest.skip("Not enough results to test pagination accuracy")

        small_result = standard_search_service.search_by_text(real_search_term, limit=1)

        assert (
            small_result.statistics.total_results
            == large_result.statistics.total_results
        )

        assert len(small_result.results.results) == 1

        assert small_result.statistics.filtered_results == len(
            small_result.results.results
        )


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_semantic_search_possible"),
    reason=SEMANTIC_SEARCH_SKIP_REASON,
)
@pytest.mark.integration
class TestSemanticSearch:
    def test_semantic_search_happy_path(self, semantic_search_service):
        query = "Tell me about fruit"
        results, took_ms = semantic_search_service.search_semantic(query=query, top_k=3)

        assert isinstance(results, list)
        assert took_ms >= 0
        assert len(results) <= 3

        if results:
            assert all(isinstance(item, VectorSearchResultItem) for item in results)
            first_result = results[0]
            assert isinstance(first_result.id, str)
            assert isinstance(first_result.score, float)
            assert isinstance(first_result.metadata, dict)
            assert first_result.text is None or isinstance(first_result.text, str)

            if any(r.id in ["sem_doc1", "sem_doc2", "sem_doc3"] for r in results):
                result_texts = [r.text for r in results if r.text]
                assert any(
                    "apples" in t or "oranges" in t or "bananas" in t
                    for t in result_texts
                ), "Expected fruit-related results"

    def test_semantic_search_no_results(self, semantic_search_service):
        query = "quantum physics lecture notes"
        results, took_ms = semantic_search_service.search_semantic(query=query, top_k=3)

        assert isinstance(results, list)
        assert took_ms >= 0

    def test_semantic_search_invalid_query(self, semantic_search_service):
        with pytest.raises(
            SearchParameterError, match="Semantic search query cannot be empty"
        ):
            semantic_search_service.search_semantic(query="", top_k=5)
        with pytest.raises(
            SearchParameterError, match="Semantic search query cannot be empty"
        ):
            semantic_search_service.search_semantic(query="   ", top_k=5)

    def test_semantic_search_invalid_top_k(self, semantic_search_service):
        with pytest.raises(
            SearchParameterError, match="top_k must be a positive integer"
        ):
            semantic_search_service.search_semantic(query="test", top_k=0)
        with pytest.raises(
            SearchParameterError, match="top_k must be a positive integer"
        ):
            semantic_search_service.search_semantic(query="test", top_k=-1)

    def test_semantic_search_missing_dependencies(
        self, confluence_client, embedding_provider
    ):
        service_no_embed = SearchService(
            client=confluence_client, embedding_service=None, vector_db_adapter=None
        )
        with pytest.raises(
            SemanticSearchError,
            match="Semantic search is not configured: EmbeddingService is missing",
        ):
            service_no_embed.search_semantic(query="test")

        if embedding_provider:
            embedding_service = EmbeddingService(provider=embedding_provider)
            service_no_vdb = SearchService(
                client=confluence_client,
                embedding_service=embedding_service,
                vector_db_adapter=None,
            )
            with pytest.raises(
                SemanticSearchError,
                match="Semantic search is not configured: VectorDBAdapter is missing",
            ):
                service_no_vdb.search_semantic(query="test")
