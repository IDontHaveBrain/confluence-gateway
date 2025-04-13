from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.models import ConfluencePage, SearchResult
from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.config import search_config
from confluence_gateway.services.search import EnhancedSearchResult, SearchService


def create_mock_search_page(page_id: str, title: str) -> MagicMock:
    page = MagicMock(spec=ConfluencePage)
    page.id = page_id
    page.title = title
    page.space = MagicMock(key="TEST")
    return page


@pytest.mark.integration
class TestSearchServiceHybridLogic:
    @pytest.fixture(autouse=True)
    def enable_hybrid_search(self):
        original_value = search_config.hybrid_search_enabled
        search_config.hybrid_search_enabled = True
        yield
        search_config.hybrid_search_enabled = original_value

    def test_hybrid_search_rrf_ranking(
        self, semantic_search_service: SearchService, mocker
    ):
        query = "test query"
        mock_kw_page1 = create_mock_search_page("page1", "Keyword Match 1")
        mock_kw_page2 = create_mock_search_page("page2", "Keyword Match 2")
        mock_kw_page_shared = create_mock_search_page("shared", "Shared Document")
        mock_keyword_result = SearchResult(
            results=[mock_kw_page_shared, mock_kw_page1, mock_kw_page2],
            total_size=3,
            limit=search_config.hybrid_keyword_fetch_limit,
            start=0,
        )
        mock_client_search = mocker.patch.object(
            semantic_search_service.client, "search", return_value=mock_keyword_result
        )
        mock_sem_item1 = VectorSearchResultItem(
            id="shared_chunk0", score=0.9, metadata={"original_content_id": "shared"}
        )
        mock_sem_item2 = VectorSearchResultItem(
            id="page3_chunk0", score=0.8, metadata={"original_content_id": "page3"}
        )
        mock_sem_item3 = VectorSearchResultItem(
            id="shared_chunk1", score=0.7, metadata={"original_content_id": "shared"}
        )
        mock_semantic_results_raw = [mock_sem_item1, mock_sem_item2, mock_sem_item3]
        mock_search_semantic = mocker.patch.object(
            semantic_search_service,
            "search_semantic",
            return_value=(mock_semantic_results_raw, 50.0),
        )
        mock_sem_page3 = create_mock_search_page("page3", "Semantic Match Only")

        def mock_get_page_side_effect(page_id, expand=None):
            if page_id == "page3":
                return mock_sem_page3
            elif page_id == "shared":
                return mock_kw_page_shared
            elif page_id == "page1":
                return mock_kw_page1
            elif page_id == "page2":
                return mock_kw_page2
            else:
                return None

        mock_client_get_page = mocker.patch.object(
            semantic_search_service.client,
            "get_page",
            side_effect=mock_get_page_side_effect,
        )
        result = semantic_search_service.search_hybrid(text=query, limit=10, start=0)
        assert isinstance(result, EnhancedSearchResult)
        mock_client_search.assert_called_once()
        mock_search_semantic.assert_called_once()
        final_results = result.results.results
        assert len(final_results) == 4
        assert final_results[0].id == "shared"
        assert final_results[1].id in ["page1", "page3"]
        assert final_results[2].id in ["page1", "page3"]
        assert final_results[1].id != final_results[2].id
        assert final_results[3].id == "page2"
        mock_client_get_page.assert_any_call(
            "page3", expand=search_config.default_expand
        )

    def test_hybrid_search_disabled(
        self, semantic_search_service: SearchService, mocker
    ):
        search_config.hybrid_search_enabled = False
        query = "test query"
        mocker.patch.object(semantic_search_service.client, "search")
        mocker.patch.object(semantic_search_service, "search_semantic")
        with pytest.raises(Exception, match="Hybrid search is disabled"):
            semantic_search_service.search_hybrid(text=query)
