"""
Index validation script.
Verifies indexes are built correctly and tests a sample retrieval.

Usage:
    python scripts/validate_index.py
    python scripts/validate_index.py --query "printed books"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def validate(config_dir: str = "config", test_query: str = "printed books dictionaries"):
    logger.info("=" * 60)
    logger.info("Commodity Code Index Validator")
    logger.info("=" * 60)

    config = ConfigLoader(config_dir)
    index_path = config.app.get("index_path", "data/indexes")

    retriever = HybridRetriever(
        retrieval_config=config.retrieval,
        embedding_config=config.embedding,
        index_path=index_path
    )

    # check indexes exist and load
    logger.info("Loading indexes...")
    loaded = retriever.load()

    if not loaded:
        logger.error("Index validation FAILED — could not load indexes")
        logger.error("Run: python scripts/build_index.py")
        sys.exit(1)

    logger.info(f"Index loaded: {retriever.index_size} records")

    # run test retrieval
    logger.info(f"\nTest query: '{test_query}'")
    logger.info("-" * 40)

    candidates, signal = retriever.retrieve(test_query)

    logger.info(f"Clustering signal: {signal['summary']}")
    logger.info(f"\nTop {len(candidates)} candidates:")

    for i, c in enumerate(candidates):
        logger.info(
            f"  {i+1}. [{c['national_code']}] "
            f"{c.get('tar_dsc', c.get('tar_all', 'N/A'))[:60]}"
        )
        logger.info(
            f"     Chapter: {c.get('hs2_dsc', 'N/A')[:50]}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Validation PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate commodity code search indexes"
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Path to config directory"
    )
    parser.add_argument(
        "--query",
        default="printed books dictionaries",
        help="Test query to run against the index"
    )
    args = parser.parse_args()
    validate(args.config_dir, args.query)
