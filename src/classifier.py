"""
Main classifier orchestrator.
Wires all components together into a single classify() call.
This is the core of the application — everything flows through here.
"""

import logging
import uuid
from datetime import datetime

from src.config.config_loader import ConfigLoader
from src.database.connector import DatabaseConnector
from src.database.query_builder import QueryBuilder
from src.indexing.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.preprocessing.pipeline import PreprocessingPipeline
from src.prompt.builder import PromptBuilder
from src.llm.client import LLMClientFactory
from src.llm.response_parser import ResponseParser
from src.history.logger import HistoryLogger

logger = logging.getLogger(__name__)


class CommodityClassifier:
    """
    Main commodity classification system.
    All components are initialized from config — swap implementations
    without touching this class.
    """

    def __init__(self, config_dir: str = "config"):
        logger.info("Initializing CommodityClassifier...")
        self.config = ConfigLoader(config_dir)

        # database
        self.db_connector = DatabaseConnector(self.config.database)
        self.query_builder = QueryBuilder(
            self.config.database,
            self.db_connector.engine
        )

        # LLM client (initialized before preprocessing — HyDE/rewriting need it)
        self.llm_client = LLMClientFactory.create(self.config.llm)

        # preprocessing pipeline
        self.preprocessor = PreprocessingPipeline(
            self.config.preprocessing,
            llm_client=self.llm_client
        )

        # retrieval
        index_path = self.config.app.get("index_path", "data/indexes")
        self.retriever = HybridRetriever(
            retrieval_config=self.config.retrieval,
            embedding_config=self.config.embedding,
            index_path=index_path
        )

        # prompt and response
        self.prompt_builder = PromptBuilder(self.config.llm)
        self.response_parser = ResponseParser()

        # history
        history_db = self.config.app.get("history_db_path", "data/history.db")
        self.history_logger = HistoryLogger(history_db)

        logger.info("CommodityClassifier initialized")

    def load(self):
        """
        Load indexes into memory.
        Called once at application startup.
        Fails loudly if indexes don't exist — run build_index.py first.
        """
        self.history_logger.initialize()
        loaded = self.retriever.load()
        if not loaded:
            raise RuntimeError(
                "Failed to load indexes. "
                "Run: python scripts/build_index.py"
            )
        logger.info("CommodityClassifier ready")

    def classify(
        self,
        query: str,
        request_id: str = None
    ) -> dict:
        """
        Classify a product description into a commodity code.

        Args:
            query: raw user product description
            request_id: optional tracking ID

        Returns:
            Classification result dict with confidence, code, reasoning etc.
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = datetime.now()

        logger.info(f"[{request_id}] Classifying: '{query[:80]}...'")

        # step 1: preprocess query
        processed_query = self.preprocessor.process(query)
        logger.debug(f"[{request_id}] Processed query: '{processed_query}'")

        # step 2: retrieve candidates + hierarchy signal
        candidates, signal = self.retriever.retrieve(processed_query)
        logger.info(
            f"[{request_id}] Retrieved {len(candidates)} candidates, "
            f"signal: {signal['confidence_level']}"
        )

        # step 3: handle no results case
        if not candidates:
            result = self._no_candidates_result()
            self._log_result(
                request_id, query, processed_query,
                result, start_time
            )
            return result

        # step 4: build prompt
        messages = self.prompt_builder.build(
            query=processed_query,
            candidates=candidates,
            signal=signal
        )

        # step 5: call LLM
        logger.debug(f"[{request_id}] Calling LLM...")
        raw_response = self.llm_client.complete(messages)
        logger.debug(f"[{request_id}] LLM responded")

        # step 6: parse response
        result = self.response_parser.parse(
            raw_response=raw_response,
            original_query=query,
            candidates=candidates
        )

        # step 7: enrich result with metadata
        processing_time = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )
        result["request_id"] = request_id
        result["timestamp"] = start_time.isoformat()
        result["original_query"] = query
        result["processed_query"] = processed_query
        result["processing_time_ms"] = processing_time

        logger.info(
            f"[{request_id}] Classification complete: "
            f"confidence={result['confidence']}, "
            f"code={result.get('national_code')}, "
            f"time={processing_time}ms"
        )

        # step 8: log to history (non-blocking failure)
        self._log_result(
            request_id, query, processed_query, result, start_time
        )

        return result

    def rebuild_index(self):
        """
        Rebuild indexes from current database data.
        Called by the /reindex API endpoint.
        """
        logger.info("Starting index rebuild from database...")
        records = self.query_builder.fetch_all_national_codes()
        self.retriever.rebuild_index(
            records=records,
            embedder=self.retriever.embedder
        )
        logger.info("Index rebuild complete")

    def _no_candidates_result(self) -> dict:
        """Return result when no candidates were retrieved."""
        return {
            "confidence": "NO MATCH",
            "national_code": None,
            "description": None,
            "reasoning": (
                "No relevant commodity codes were retrieved for this description. "
                "The description may be too vague, use very non-standard terminology, "
                "or the product may not be covered by available codes."
            ),
            "hierarchy_path": None,
            "alternatives": None,
            "exclusions_noted": None
        }

    def _log_result(
        self,
        request_id: str,
        original_query: str,
        processed_query: str,
        result: dict,
        start_time: datetime
    ):
        """Log classification result to history — never raises."""
        try:
            processing_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            self.history_logger.log({
                "request_id": request_id,
                "timestamp": start_time.isoformat(),
                "original_query": original_query,
                "processed_query": processed_query,
                "processing_time_ms": processing_time,
                **result
            })
        except Exception as e:
            logger.error(f"History logging failed: {e}")
