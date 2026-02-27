"""
Database connection handler.
Supports any SQLAlchemy-compatible database.
Connection string is loaded from environment variable.
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseConnector:
    def __init__(self, config: dict):
        self.config = config
        self._engine: Optional[Engine] = None

    def connect(self) -> Engine:
        """Create and return database engine."""
        if self._engine is not None:
            return self._engine

        connection_string = self.config.get("connection_string")
        if not connection_string:
            raise ValueError(
                "Database connection string not configured. "
                "Set DB_CONNECTION_STRING environment variable."
            )

        logger.info("Connecting to database...")
        self._engine = create_engine(
            connection_string,
            pool_pre_ping=True,   # verify connection before use
            pool_recycle=3600     # recycle connections every hour
        )

        # verify connection works
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established")

        return self._engine

    def disconnect(self):
        """Close database connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            return self.connect()
        return self._engine
