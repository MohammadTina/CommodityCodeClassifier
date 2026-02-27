"""
Text embedding module.
Converts commodity code records into vector representations.
Embedding model is fully configurable — swap via config/embedding.yaml.
"""

import logging
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "BAAI/bge-large-en-v1.5")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize_embeddings", True)
        self.text_template = config.get(
            "text_template",
            "{tar_all} {hs6_dsc} {hs4_dsc} {hs2_dsc}"
        )
        self._model: Optional[SentenceTransformer] = None

    def load(self):
        """Load embedding model into memory."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        logger.info("Embedding model loaded")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.load()
        return self._model

    def build_text(self, record: dict) -> str:
        """
        Build embeddable text from a commodity code record.
        Uses template from config — fields must exist in record dict.
        Falls back gracefully if a field is missing or None.
        """
        # replace None values with empty string before formatting
        safe_record = {
            k: (v if v is not None else "")
            for k, v in record.items()
        }
        try:
            return self.text_template.format(**safe_record).strip()
        except KeyError as e:
            logger.warning(f"Template field missing in record: {e}")
            # fallback to basic description fields
            parts = [
                safe_record.get("tar_all", ""),
                safe_record.get("hs6_dsc", ""),
                safe_record.get("hs4_dsc", ""),
            ]
            return " ".join(p for p in parts if p).strip()

    def embed_records(self, records: list[dict]) -> np.ndarray:
        """
        Embed a list of commodity code records.
        Returns numpy array of shape (n_records, embedding_dim).
        """
        texts = [self.build_text(r) for r in records]
        logger.info(f"Embedding {len(texts)} records in batches of {self.batch_size}...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query for retrieval.
        BGE models benefit from a query instruction prefix.
        """
        # BGE models use this prefix for retrieval queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize
        )
        return embedding[0]

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension for FAISS index initialization."""
        return self.model.get_sentence_embedding_dimension()
