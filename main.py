"""
Application entry point.
Run with: python main.py
Or production: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
from pathlib import Path

import uvicorn

# ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.api import app

# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data/logs/app.log")
        ]
    )
    # suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


if __name__ == "__main__":
    # load log level from config if possible
    try:
        import yaml
        with open("config/app.yaml") as f:
            app_config = yaml.safe_load(f)
        log_level = app_config.get("app", {}).get("log_level", "INFO")
    except Exception:
        log_level = "INFO"

    setup_logging(log_level)

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # disable reload in production
        log_level="info"
    )
