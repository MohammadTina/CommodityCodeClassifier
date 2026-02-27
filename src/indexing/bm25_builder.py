"""
BM25 keyword search index builder.
Complements semantic search by catching exact/near-exact term matches.
Uses rank_bm25 library — lightweight and fast for 13K records.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Builder:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._bm25: Optional[BM25Okapi] = None
        self._metadata: list[dict] = []
        self._corpus_texts: list[str] = []

        self._bm25_file = self.index_path / "bm25.pkl"

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for BM25.
        Lowercases and splits on non-alphanumeric characters.
        """
        if not text:
            return []
        text = text.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        return [t for t in tokens if t]

    def build(self, texts: list[str], metadata: list[dict]):
        """
        Build BM25 index from text corpus.

        Args:
            texts: list of text strings (one per commodity code)
            metadata: list of metadata dicts (same order as texts)
        """
        logger.info(f"Building BM25 index for {len(texts)} documents")
        self._corpus_texts = texts
        self._metadata = metadata

        tokenized_corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built")

    def save(self):
        """Persist BM25 index to disk."""
        if self._bm25 is None:
            raise RuntimeError("No BM25 index to save — build first")

        data = {
            "bm25": self._bm25,
            "metadata": self._metadata,
            "corpus_texts": self._corpus_texts
        }
        with open(self._bm25_file, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved")

    def load(self) -> bool:
        """
        Load BM25 index from disk.
        Returns True if loaded successfully, False if not found.
        """
        if not self._bm25_file.exists():
            logger.warning("No BM25 index found — run build_index.py first")
            return False

        with open(self._bm25_file, "rb") as f:
            data = pickle.load(f)

        self._bm25 = data["bm25"]
        self._metadata = data["metadata"]
        self._corpus_texts = data["corpus_texts"]
        logger.info(f"BM25 index loaded: {len(self._metadata)} documents")
        return True

    def search(self, query: str, top_k: int) -> list[dict]:
        """
        Search BM25 index for best keyword matches.

        Args:
            query: raw query string
            top_k: number of results to return

        Returns:
            List of metadata dicts sorted by BM25 score
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded — call load() first")

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # get top_k indices by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue  # skip zero-score results
            result = self._metadata[idx].copy()
            result["_bm25_score"] = float(scores[idx])
            results.append(result)

        return results

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None
