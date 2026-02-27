"""
Classification history logger.
Stores all classification results in SQLite for auditing and improvement.
Uses async SQLite to avoid blocking the API during logging.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HistoryLogger:
    """
    Logs classification results to SQLite.
    Sync implementation — simple and reliable for the request volumes expected.
    """

    def __init__(self, db_path: str = "data/history.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    request_id         TEXT PRIMARY KEY,
                    timestamp          TEXT NOT NULL,
                    original_query     TEXT NOT NULL,
                    processed_query    TEXT,
                    confidence         TEXT,
                    national_code      TEXT,
                    description        TEXT,
                    reasoning          TEXT,
                    hierarchy_path     TEXT,
                    alternatives       TEXT,
                    exclusions_noted   TEXT,
                    processing_time_ms INTEGER,
                    created_at         TEXT DEFAULT (datetime('now'))
                )
            """)
            # index for common query patterns
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_confidence
                ON classifications(confidence)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_national_code
                ON classifications(national_code)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON classifications(timestamp)
            """)
            conn.commit()
        logger.info(f"History database initialized: {self.db_path}")

    def log(self, record: dict):
        """
        Log a classification result.

        Args:
            record: dict containing all classification fields
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO classifications (
                        request_id, timestamp, original_query,
                        processed_query, confidence, national_code,
                        description, reasoning, hierarchy_path,
                        alternatives, exclusions_noted, processing_time_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    record.get("request_id"),
                    record.get("timestamp", datetime.now().isoformat()),
                    record.get("original_query"),
                    record.get("processed_query"),
                    record.get("confidence"),
                    record.get("national_code"),
                    record.get("description"),
                    record.get("reasoning"),
                    json.dumps(record.get("hierarchy_path")),
                    json.dumps(record.get("alternatives")),
                    record.get("exclusions_noted"),
                    record.get("processing_time_ms")
                ))
                conn.commit()
        except Exception as e:
            # logging failure should never break classification
            logger.error(f"Failed to log classification: {e}")

    def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        confidence_filter: Optional[str] = None,
        national_code_filter: Optional[str] = None
    ) -> dict:
        """
        Retrieve paginated classification history.

        Args:
            page: page number (1-indexed)
            page_size: results per page
            confidence_filter: filter by HIGH | LOW | NO MATCH
            national_code_filter: filter by specific national code

        Returns:
            Dict with total count, page info, and results list
        """
        offset = (page - 1) * page_size
        where_clauses = []
        params = []

        if confidence_filter:
            where_clauses.append("confidence = ?")
            params.append(confidence_filter)
        if national_code_filter:
            where_clauses.append("national_code = ?")
            params.append(national_code_filter)

        where_sql = (
            "WHERE " + " AND ".join(where_clauses)
            if where_clauses else ""
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # total count
            total = conn.execute(
                f"SELECT COUNT(*) FROM classifications {where_sql}",
                params
            ).fetchone()[0]

            # paginated results
            rows = conn.execute(
                f"""SELECT * FROM classifications {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?""",
                params + [page_size, offset]
            ).fetchall()

        results = [self._row_to_dict(row) for row in rows]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "results": results
        }

    def get_by_id(self, request_id: str) -> Optional[dict]:
        """Retrieve a single classification by request ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM classifications WHERE request_id = ?",
                (request_id,)
            ).fetchone()

        return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row) -> dict:
        """Convert SQLite row to dict, deserializing JSON fields."""
        d = dict(row)
        for json_field in ("hierarchy_path", "alternatives"):
            if d.get(json_field):
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    d[json_field] = None
        return d
