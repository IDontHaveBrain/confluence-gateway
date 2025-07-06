import pytest
from unittest.mock import patch

from confluence_gateway.services.ranking import reciprocal_rank_fusion


class TestRankingService:
    def test_rrf_basic_fusion(self):
        """Test basic fusion of keyword and semantic results."""
        keyword_ids = ["doc1", "doc2", "doc3"]
        semantic_results = [("doc2", 0.9), ("doc4", 0.8), ("doc1", 0.7)]
        k = 60

        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        result_ids = [doc_id for doc_id, _ in results]
        assert set(result_ids) == {"doc1", "doc2", "doc3", "doc4"}

        assert results[0][0] == "doc2"
        assert results[1][0] == "doc1"
        assert abs(results[0][1] - 0.0325) < 0.0001
        assert abs(results[1][1] - 0.0323) < 0.0001

    def test_rrf_different_k_values(self):
        """Test RRF with different k values."""
        keyword_ids = ["doc1", "doc2"]
        semantic_results = [("doc1", 0.9), ("doc3", 0.8)]

        results_k1 = reciprocal_rank_fusion(keyword_ids, semantic_results, k=1)
        assert results_k1[0][0] == "doc1"
        assert abs(results_k1[0][1] - 1.0) < 0.001

        results_k10 = reciprocal_rank_fusion(keyword_ids, semantic_results, k=10)
        assert results_k10[0][0] == "doc1"
        assert abs(results_k10[0][1] - 0.182) < 0.001

        results_k100 = reciprocal_rank_fusion(keyword_ids, semantic_results, k=100)
        assert results_k100[0][0] == "doc1"
        assert abs(results_k100[0][1] - 0.0198) < 0.001

    def test_rrf_overlapping_results(self):
        """Test RRF with heavy overlap between result sets."""
        keyword_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        semantic_results = [
            ("doc3", 0.9),
            ("doc1", 0.8),
            ("doc4", 0.7),
            ("doc2", 0.6),
            ("doc5", 0.5),
        ]
        k = 60

        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        result_ids = [doc_id for doc_id, _ in results]
        assert len(result_ids) == 5
        assert set(result_ids) == {"doc1", "doc2", "doc3", "doc4", "doc5"}

        for doc_id, score in results:
            assert score > 0

    def test_rrf_empty_lists(self):
        """Test RRF with empty input lists."""
        results = reciprocal_rank_fusion([], [], k=60)
        assert results == []

        semantic_results = [("doc1", 0.9), ("doc2", 0.8)]
        results = reciprocal_rank_fusion([], semantic_results, k=60)
        assert len(results) == 2
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

        keyword_ids = ["doc1", "doc2"]
        results = reciprocal_rank_fusion(keyword_ids, [], k=60)
        assert len(results) == 2
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"

    def test_rrf_single_source(self):
        """Test RRF when only one source has results."""
        keyword_ids = ["doc1", "doc2", "doc3"]
        results = reciprocal_rank_fusion(keyword_ids, [], k=60)

        assert len(results) == 3
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"
        assert results[2][0] == "doc3"
        assert results[0][1] > results[1][1] > results[2][1]

        semantic_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        results = reciprocal_rank_fusion([], semantic_results, k=60)

        assert len(results) == 3
        assert results[0][0] == "doc1"
        assert results[1][0] == "doc2"
        assert results[2][0] == "doc3"

    def test_rrf_invalid_k_value(self):
        """Test RRF with invalid k values."""
        keyword_ids = ["doc1"]
        semantic_results = [("doc1", 0.9)]

        with pytest.raises(ValueError, match="RRF constant k must be positive"):
            reciprocal_rank_fusion(keyword_ids, semantic_results, k=0)

        with pytest.raises(ValueError, match="RRF constant k must be positive"):
            reciprocal_rank_fusion(keyword_ids, semantic_results, k=-1)

        with pytest.raises(ValueError, match="RRF constant k must be positive"):
            reciprocal_rank_fusion(keyword_ids, semantic_results, k=-100)

    def test_rrf_large_result_sets(self):
        """Test RRF with large result sets."""
        keyword_ids = [f"doc{i}" for i in range(1000)]

        semantic_results = [(f"doc{i}", 1.0 - i / 1000) for i in range(500, 1500)]

        k = 60
        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        assert len(results) == 1500

        for doc_id, score in results:
            assert score > 0

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_preservation(self):
        """Test that RRF preserves relative rankings appropriately."""
        keyword_ids = ["doc1", "doc2", "doc3", "doc4"]
        semantic_results = [
            ("doc1", 0.95),
            ("doc5", 0.9),
            ("doc2", 0.85),
            ("doc6", 0.8),
        ]

        k = 60
        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        assert results[0][0] == "doc1"

        assert results[1][0] == "doc2"

        result_ids = [doc_id for doc_id, _ in results]
        assert len(result_ids) == len(set(result_ids))

    def test_rrf_duplicate_handling(self):
        """Test RRF handles duplicate document IDs correctly."""
        keyword_ids = ["doc1", "doc2", "doc1"]
        semantic_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        k = 60
        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        result_ids = [doc_id for doc_id, _ in results]
        assert len(result_ids) == len(set(result_ids))
        assert set(result_ids) == {"doc1", "doc2", "doc3"}

    def test_rrf_semantic_sorting(self):
        """Test that semantic results are properly sorted by score before ranking."""
        semantic_results = [("doc3", 0.5), ("doc1", 0.9), ("doc2", 0.7)]
        keyword_ids = ["doc4"]

        k = 60
        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        doc1_score = next(score for doc_id, score in results if doc_id == "doc1")
        doc2_score = next(score for doc_id, score in results if doc_id == "doc2")
        doc3_score = next(score for doc_id, score in results if doc_id == "doc3")

        assert doc1_score > doc2_score > doc3_score

    @patch("confluence_gateway.services.ranking.logger")
    def test_rrf_logging(self, mock_logger):
        """Test that RRF logs debug information correctly."""
        keyword_ids = ["doc1", "doc2"]
        semantic_results = [("doc2", 0.9), ("doc3", 0.8)]
        k = 60

        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        assert mock_logger.debug.call_count == 2

        first_call = mock_logger.debug.call_args_list[0][0][0]
        assert "RRF: Processing 3 unique IDs" in first_call
        assert "2 keyword and 2 semantic results" in first_call
        assert "k=60" in first_call

        second_call = mock_logger.debug.call_args_list[1][0][0]
        assert "RRF: Produced 3 ranked results" in second_call

    def test_rrf_edge_case_single_document(self):
        """Test RRF with a single document in both result sets."""
        keyword_ids = ["doc1"]
        semantic_results = [("doc1", 0.95)]
        k = 60

        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        assert len(results) == 1
        assert results[0][0] == "doc1"
        expected_score = 2.0 / 61.0
        assert abs(results[0][1] - expected_score) < 0.0001

    def test_rrf_no_overlap(self):
        """Test RRF with completely disjoint result sets."""
        keyword_ids = ["doc1", "doc2", "doc3"]
        semantic_results = [("doc4", 0.9), ("doc5", 0.8), ("doc6", 0.7)]
        k = 60

        results = reciprocal_rank_fusion(keyword_ids, semantic_results, k)

        assert len(results) == 6
        result_ids = [doc_id for doc_id, _ in results]
        assert set(result_ids) == {"doc1", "doc2", "doc3", "doc4", "doc5", "doc6"}

        doc1_score = next(score for doc_id, score in results if doc_id == "doc1")
        doc6_score = next(score for doc_id, score in results if doc_id == "doc6")
        assert doc1_score > doc6_score
