"""
FAISS vector index builder and manager.
Builds, saves, loads and searches the vector index.
At 13K records this runs entirely in memory — fast and simple.
"""

import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, config: dict, index_path: str):
        self.config = config
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: list[dict] = []

        # file paths for persistence
        self._index_file = self.index_path / "faiss.index"
        self._metadata_file = self.index_path / "metadata.pkl"

    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        Build FAISS index from embeddings and store metadata.

        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: list of dicts, one per embedding, containing
                      all commodity code fields for retrieval
        """
        n, dim = embeddings.shape
        logger.info(f"Building FAISS index: {n} records, {dim} dimensions")

        # Inner product index (works with normalized embeddings = cosine similarity)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._metadata = metadata

        logger.info(f"FAISS index built with {self._index.ntotal} vectors")

    def save(self):
        """Persist index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save — build index first")

        logger.info(f"Saving index to {self.index_path}")
        faiss.write_index(self._index, str(self._index_file))
        with open(self._metadata_file, "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info("Index saved successfully")

    def load(self) -> bool:
        """
        Load index and metadata from disk.
        Returns True if loaded successfully, False if not found.
        """
        if not self._index_file.exists() or not self._metadata_file.exists():
            logger.warning("No saved index found — run build_index.py first")
            return False

        logger.info(f"Loading FAISS index from {self.index_path}")
        self._index = faiss.read_index(str(self._index_file))
        with open(self._metadata_file, "rb") as f:
            self._metadata = pickle.load(f)

        logger.info(f"Index loaded: {self._index.ntotal} vectors")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """
        Search index for nearest neighbors.

        Args:
            query_embedding: 1D numpy array
            top_k: number of results to return

        Returns:
            List of metadata dicts sorted by similarity score
        """
        if self._index is None:
            raise RuntimeError("Index not loaded — call load() first")

        # FAISS expects 2D array
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            result = self._metadata[idx].copy()
            result["_semantic_score"] = float(score)
            results.append(result)

        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal
