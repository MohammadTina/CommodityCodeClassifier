"""
Hierarchy clustering analyzer.
Analyzes retrieved candidates to produce a confidence signal
based on how tightly they cluster around specific hierarchy levels.

Key insight: if 8/10 candidates share the same HS4 heading,
that's a strong confidence signal for the LLM.
If they scatter across 5 different HS2 chapters, that's uncertainty.
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


class HierarchyAnalyzer:
    def __init__(self, config: dict):
        self.high_threshold = config.get("confidence", {}).get(
            "high_threshold", 0.7
        )
        self.low_threshold = config.get("confidence", {}).get(
            "low_threshold", 0.4
        )

    def analyze(self, candidates: list[dict]) -> dict:
        """
        Analyze hierarchy clustering across retrieved candidates.

        Returns a signal dict containing:
        - confidence_level: HIGH | MEDIUM | LOW
        - dominant_hs2: most common chapter code
        - dominant_hs4: most common heading code
        - hs2_concentration: ratio of candidates in dominant chapter
        - hs4_concentration: ratio of candidates in dominant heading
        - summary: human readable signal for LLM prompt
        """
        if not candidates:
            return self._empty_signal()

        n = len(candidates)

        # count distribution across hierarchy levels
        hs2_counts = Counter(c.get("hs2_cod", "") for c in candidates)
        hs4_counts = Counter(c.get("hs4_cod", "") for c in candidates)
        hs6_counts = Counter(c.get("hs6_cod", "") for c in candidates)

        # get dominant codes and their concentration ratios
        dominant_hs2, hs2_top_count = hs2_counts.most_common(1)[0]
        dominant_hs4, hs4_top_count = hs4_counts.most_common(1)[0]
        dominant_hs6, hs6_top_count = hs6_counts.most_common(1)[0]

        hs2_concentration = hs2_top_count / n
        hs4_concentration = hs4_top_count / n
        hs6_concentration = hs6_top_count / n

        # determine confidence level
        # use HS4 concentration as primary signal — heading level is most useful
        if hs4_concentration >= self.high_threshold:
            confidence_level = "HIGH"
        elif hs4_concentration >= self.low_threshold:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # get descriptions for dominant codes from candidates
        dominant_hs2_dsc = next(
            (c.get("hs2_dsc", "") for c in candidates
             if c.get("hs2_cod") == dominant_hs2), ""
        )
        dominant_hs4_dsc = next(
            (c.get("hs4_dsc", "") for c in candidates
             if c.get("hs4_cod") == dominant_hs4), ""
        )

        # build human readable summary for LLM prompt
        summary = self._build_summary(
            n=n,
            confidence_level=confidence_level,
            hs2_top_count=hs2_top_count,
            hs4_top_count=hs4_top_count,
            dominant_hs2=dominant_hs2,
            dominant_hs4=dominant_hs4,
            dominant_hs2_dsc=dominant_hs2_dsc,
            dominant_hs4_dsc=dominant_hs4_dsc
        )

        return {
            "confidence_level": confidence_level,
            "dominant_hs2": dominant_hs2,
            "dominant_hs2_dsc": dominant_hs2_dsc,
            "dominant_hs4": dominant_hs4,
            "dominant_hs4_dsc": dominant_hs4_dsc,
            "hs2_concentration": round(hs2_concentration, 2),
            "hs4_concentration": round(hs4_concentration, 2),
            "hs6_concentration": round(hs6_concentration, 2),
            "total_candidates": n,
            "summary": summary
        }

    def _build_summary(
        self,
        n: int,
        confidence_level: str,
        hs2_top_count: int,
        hs4_top_count: int,
        dominant_hs2: str,
        dominant_hs4: str,
        dominant_hs2_dsc: str,
        dominant_hs4_dsc: str
    ) -> str:
        if confidence_level == "HIGH":
            return (
                f"{hs4_top_count}/{n} candidates cluster around "
                f"heading {dominant_hs4} ({dominant_hs4_dsc}) — "
                f"strong signal for classification"
            )
        elif confidence_level == "MEDIUM":
            return (
                f"{hs4_top_count}/{n} candidates in heading {dominant_hs4} "
                f"({dominant_hs4_dsc}), {hs2_top_count}/{n} in chapter "
                f"{dominant_hs2} ({dominant_hs2_dsc}) — moderate signal"
            )
        else:
            return (
                f"Candidates scattered across multiple headings "
                f"(strongest cluster: {hs4_top_count}/{n} in heading "
                f"{dominant_hs4}) — low confidence signal"
            )

    def _empty_signal(self) -> dict:
        return {
            "confidence_level": "LOW",
            "dominant_hs2": None,
            "dominant_hs2_dsc": None,
            "dominant_hs4": None,
            "dominant_hs4_dsc": None,
            "hs2_concentration": 0.0,
            "hs4_concentration": 0.0,
            "hs6_concentration": 0.0,
            "total_candidates": 0,
            "summary": "No candidates retrieved"
        }
