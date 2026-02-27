"""
Index builder script.
Run this ONCE before starting the API, and again whenever commodity codes change.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --config-dir config
"""

import argparse
import logging
import sys
from pathlib import Path

# ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.database.connector import DatabaseConnector
from src.database.query_builder import QueryBuilder
from src.indexing.embedder import Embedder
from src.indexing.index_builder import IndexBuilder
from src.indexing.bm25_builder import BM25Builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def build_index(config_dir: str = "config"):
    logger.info("=" * 60)
    logger.info("Commodity Code Index Builder")
    logger.info("=" * 60)

    # load config
    config = ConfigLoader(config_dir)
    index_path = config.app.get("index_path", "data/indexes")

    # connect to database
    logger.info("Step 1/5: Connecting to database...")
    db_connector = DatabaseConnector(config.database)
    engine = db_connector.engine

    # fetch all national codes with full hierarchy
    logger.info("Step 2/5: Fetching commodity codes from database...")
    query_builder = QueryBuilder(config.database, engine)
    records = query_builder.fetch_all_national_codes()

    if not records:
        logger.error("No records fetched from database. Check connection and config.")
        sys.exit(1)

    logger.info(f"Fetched {len(records)} active national commodity codes")

    # load embedding model
    logger.info("Step 3/5: Loading embedding model...")
    embedder = Embedder(config.embedding)
    embedder.load()

    # build embeddable texts
    texts = [embedder.build_text(r) for r in records]

    # log sample to verify text construction
    logger.info("Sample embedded text:")
    logger.info(f"  {texts[0][:150]}...")

    # build and save FAISS index
    logger.info("Step 4/5: Building and saving FAISS vector index...")
    embeddings = embedder.embed_records(records)
    faiss_index = IndexBuilder(config.embedding, index_path)
    faiss_index.build(embeddings, records)
    faiss_index.save()
    logger.info(f"FAISS index saved: {faiss_index.size} vectors")

    # build and save BM25 index
    logger.info("Step 5/5: Building and saving BM25 keyword index...")
    bm25_index = BM25Builder(index_path)
    bm25_index.build(texts, records)
    bm25_index.save()

    # disconnect
    db_connector.disconnect()

    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info(f"  Records indexed : {len(records)}")
    logger.info(f"  Index location  : {index_path}")
    logger.info("You can now start the API: python main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build commodity code search indexes"
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Path to config directory (default: config)"
    )
    args = parser.parse_args()
    build_index(args.config_dir)
