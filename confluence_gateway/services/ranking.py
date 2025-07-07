import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    keyword_result_ids: list[str],
    semantic_results: list[tuple[str, float]],
    k: int,
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("RRF constant k must be positive")

    rrf_scores: dict[str, float] = {}

    keyword_ranks: dict[str, int] = {
        doc_id: rank for rank, doc_id in enumerate(keyword_result_ids, 1)
    }

    semantic_ranks: dict[str, int] = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(
            sorted(semantic_results, key=lambda item: item[1], reverse=True), 1
        )
    }

    all_doc_ids = set(keyword_ranks.keys()) | set(semantic_ranks.keys())

    logger.debug(
        f"RRF: Processing {len(all_doc_ids)} unique IDs from {len(keyword_result_ids)} keyword and {len(semantic_results)} semantic results with k={k}"
    )

    for doc_id in all_doc_ids:
        score = 0.0
        rank_kw = keyword_ranks.get(doc_id)
        if rank_kw is not None:
            score += 1.0 / (k + rank_kw)

        rank_sem = semantic_ranks.get(doc_id)
        if rank_sem is not None:
            score += 1.0 / (k + rank_sem)

        if score > 0:
            rrf_scores[doc_id] = score

    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    logger.debug(f"RRF: Produced {len(sorted_results)} ranked results.")
    return sorted_results
