"""
Reciprocal Rank Fusion (RRF) for combining semantic and BM25 results.
Simple, effective, no ML model needed.
A code ranking high in both lists gets significantly boosted.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RRFFusion:
    def __init__(self, config: dict):
        self.k = config.get("rrf_k", 60)
        self.top_k = config.get("fusion_top_k", 10)

    def fuse(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict]
    ) -> list[dict]:
        """
        Merge and re-rank results from semantic and BM25 search.

        RRF score formula:
            score(d) = sum(1 / (k + rank(d)))
        where k is a constant (default 60) that dampens the impact
        of very high rankings.

        Args:
            semantic_results: ranked list from vector search
            bm25_results: ranked list from BM25 search

        Returns:
            Merged and re-ranked list of top candidates
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        # store full record for each code
        records: dict[str, dict] = {}

        # score semantic results
        for rank, result in enumerate(semantic_results):
            code = result["national_code"]
            rrf_scores[code] += 1.0 / (self.k + rank + 1)
            if code not in records:
                records[code] = result

        # score BM25 results
        for rank, result in enumerate(bm25_results):
            code = result["national_code"]
            rrf_scores[code] += 1.0 / (self.k + rank + 1)
            if code not in records:
                records[code] = result

        # sort by fused score
        sorted_codes = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        # build final result list
        fused = []
        for code, score in sorted_codes:
            result = records[code].copy()
            result["_rrf_score"] = score
            # flag whether it appeared in both lists
            in_semantic = any(
                r["national_code"] == code for r in semantic_results
            )
            in_bm25 = any(
                r["national_code"] == code for r in bm25_results
            )
            result["_in_both"] = in_semantic and in_bm25
            fused.append(result)

        logger.debug(
            f"RRF fusion: {len(semantic_results)} semantic + "
            f"{len(bm25_results)} BM25 → {len(fused)} fused"
        )
        return fused
