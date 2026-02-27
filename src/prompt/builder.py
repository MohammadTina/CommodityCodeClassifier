"""
Prompt builder for the commodity classification LLM call.
Assembles system prompt + context block + user query.
Designed to be lean and effective for a 7B parameter model.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Legal note keywords — only surface notes containing these
LEGAL_NOTE_KEYWORDS = [
    "does not cover",
    "excluding",
    "includes",
    "applies to",
    "limited to",
    "shall not",
    "except",
    "other than",
    "not including"
]

# Max characters of legal notes to include — keeps context lean for 7B
MAX_LEGAL_NOTE_CHARS = 400


class PromptBuilder:
    def __init__(self, config: dict, template_dir: str = None):
        self.config = config
        template_dir = template_dir or "src/prompt/templates"
        self.system_prompt = self._load_template(
            Path(template_dir) / "system.txt"
        )

    def _load_template(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text().strip()

    def build(
        self,
        query: str,
        candidates: list[dict],
        signal: dict
    ) -> list[dict]:
        """
        Build the full message list for the LLM.

        Args:
            query: preprocessed user query
            candidates: top N commodity code candidates with metadata
            signal: hierarchy clustering signal from HierarchyAnalyzer

        Returns:
            List of message dicts for chat completion API
        """
        context_block = self._build_context_block(candidates, signal)
        user_message = self._build_user_message(query, context_block)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

    def _build_context_block(
        self,
        candidates: list[dict],
        signal: dict
    ) -> str:
        separator = "=" * 55
        thin_sep = "-" * 55

        signal_line = f"RETRIEVAL SIGNAL: {signal.get('summary', 'N/A')}"

        candidates_text = ""
        for rank, candidate in enumerate(candidates):
            candidates_text += self._format_candidate(candidate, rank + 1, thin_sep)

        return (
            f"CLASSIFICATION CONTEXT:\n"
            f"{separator}\n"
            f"{signal_line}\n"
            f"{candidates_text}"
            f"{separator}"
        )

    def _format_candidate(
        self,
        candidate: dict,
        rank: int,
        separator: str
    ) -> str:
        # determine rank label
        label = "Best Match" if rank == 1 else f"Rank {rank}"

        # build legal notes section — only if relevant
        legal_notes = self._extract_relevant_notes(
            candidate.get("hs2_txt", ""),
            candidate.get("hs1_txt", "")
        )
        notes_line = f"\nLegal Notes   : {legal_notes}" if legal_notes else ""

        return (
            f"\n{separator}\n"
            f"CANDIDATE {rank} [{label}]\n"
            f"{separator}\n"
            f"National Code : {candidate.get('national_code', 'N/A')}\n"
            f"Description   : {candidate.get('tar_all') or candidate.get('tar_dsc', 'N/A')}\n"
            f"Subheading    : {candidate.get('hs6_dsc', 'N/A')}\n"
            f"Heading       : {candidate.get('hs4_dsc', 'N/A')}\n"
            f"Chapter       : {candidate.get('hs2_dsc', 'N/A')}\n"
            f"Section       : {candidate.get('hs1_dsc', 'N/A')}"
            f"{notes_line}\n"
        )

    def _build_user_message(self, query: str, context_block: str) -> str:
        return (
            f"{context_block}\n\n"
            f"PRODUCT DESCRIPTION TO CLASSIFY:\n"
            f'"{query}"\n\n'
            f"Provide your classification following the required output format."
        )

    def _extract_relevant_notes(self, *texts: str) -> str:
        """
        Only surface legal notes that contain exclusion/inclusion language.
        Keeps context lean for 7B model — no dumping entire legal text.
        """
        relevant_snippets = []

        for text in texts:
            if not text:
                continue
            text_lower = text.lower()
            if any(kw in text_lower for kw in LEGAL_NOTE_KEYWORDS):
                # truncate to keep context window manageable
                snippet = text[:MAX_LEGAL_NOTE_CHARS]
                if len(text) > MAX_LEGAL_NOTE_CHARS:
                    snippet += "..."
                relevant_snippets.append(snippet)

        return " | ".join(relevant_snippets) if relevant_snippets else ""
