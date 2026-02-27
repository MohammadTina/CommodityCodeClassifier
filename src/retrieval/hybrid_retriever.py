"""
Hybrid retriever orchestrator.
Combines semantic search (FAISS) + keyword search (BM25) via RRF fusion.
This is the main retrieval interface used by the classifier.
"""

import logging
from pathlib import Path

from src.indexing.embedder import Embedder
from src.indexing.index_builder import IndexBuilder
from src.indexing.bm25_builder import BM25Builder
from src.retrieval.fusion import RRFFusion
from src.retrieval.hierarchy_analyzer import HierarchyAnalyzer

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        retrieval_config: dict,
        embedding_config: dict,
        index_path: str
    ):
        self.config = retrieval_config
        self.index_path = index_path

        # initialize components
        self.embedder = Embedder(embedding_config)
        self.faiss_index = IndexBuilder(embedding_config, index_path)
        self.bm25_index = BM25Builder(index_path)
        self.fusion = RRFFusion(retrieval_config)
        self.analyzer = HierarchyAnalyzer(retrieval_config)

        # retrieval parameters
        self.semantic_top_k = retrieval_config.get("semantic_top_k", 20)
        self.bm25_top_k = retrieval_config.get("bm25_top_k", 20)
        self.llm_candidates = retrieval_config.get("llm_candidates", 3)

    def load(self) -> bool:
        """
        Load both indexes from disk.
        Called at application startup.
        Returns True if both loaded successfully.
        """
        faiss_ok = self.faiss_index.load()
        bm25_ok = self.bm25_index.load()

        if not faiss_ok or not bm25_ok:
            logger.error(
                "Failed to load indexes. Run scripts/build_index.py first."
            )
            return False

        logger.info("Both indexes loaded successfully")
        return True

    def retrieve(self, query: str) -> tuple[list[dict], dict]:
        """
        Full retrieval pipeline for a query.

        Args:
            query: preprocessed user query

        Returns:
            Tuple of:
            - candidates: top N results after fusion (for LLM)
            - signal: hierarchy clustering analysis
        """
        # step 1: semantic search
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.faiss_index.search(
            query_embedding,
            self.semantic_top_k
        )
        logger.debug(f"Semantic search returned {len(semantic_results)} results")

        # step 2: BM25 keyword search
        bm25_results = self.bm25_index.search(
            query,
            self.bm25_top_k
        )
        logger.debug(f"BM25 search returned {len(bm25_results)} results")

        # step 3: RRF fusion
        fused_results = self.fusion.fuse(semantic_results, bm25_results)
        logger.debug(f"After fusion: {len(fused_results)} candidates")

        # step 4: hierarchy clustering analysis
        signal = self.analyzer.analyze(fused_results)
        logger.debug(f"Clustering signal: {signal['confidence_level']}")

        # step 5: trim to LLM candidates
        top_candidates = fused_results[:self.llm_candidates]

        return top_candidates, signal

    def rebuild_index(
        self,
        records: list[dict],
        embedder: Embedder = None
    ):
        """
        Rebuild both indexes from fresh database records.
        Called by the /reindex endpoint when commodity codes change.

        Args:
            records: list of commodity code dicts from database
            embedder: optional embedder override
        """
        embedder = embedder or self.embedder

        logger.info(f"Rebuilding indexes for {len(records)} records")

        # build embeddable texts
        texts = [embedder.build_text(r) for r in records]

        # build and save FAISS index
        import numpy as np
        embeddings = embedder.embed_records(records)
        self.faiss_index.build(embeddings, records)
        self.faiss_index.save()

        # build and save BM25 index
        self.bm25_index.build(texts, records)
        self.bm25_index.save()

        logger.info("Index rebuild complete")

    @property
    def is_loaded(self) -> bool:
        return self.faiss_index.is_loaded and self.bm25_index.is_loaded

    @property
    def index_size(self) -> int:
        return self.faiss_index.size
