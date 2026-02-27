"""
FastAPI REST API layer.
Exposes the classifier as HTTP endpoints.
Auto-generates OpenAPI docs at /docs.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.classifier import CommodityClassifier

logger = logging.getLogger(__name__)

# ─── Global classifier instance ──────────────────────────────────────────────
# Single instance shared across all requests
classifier: Optional[CommodityClassifier] = None


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize classifier on startup, cleanup on shutdown."""
    global classifier

    logger.info("Starting Commodity Code Classifier API...")
    classifier = CommodityClassifier(config_dir="config")
    classifier.load()
    logger.info("API ready to serve requests")

    yield

    logger.info("Shutting down...")
    if classifier and classifier.db_connector:
        classifier.db_connector.disconnect()


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Commodity Code Classifier",
    description=(
        "RAG-based commodity code classification API. "
        "Submit product descriptions and receive national commodity code recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow all origins for internal use, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class ClassificationRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Product description to classify into a commodity code",
        examples=["Dictionaries and encyclopaedias, printed"]
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional caller-provided ID for tracking. Auto-generated if not provided."
    )


class HierarchyPath(BaseModel):
    section: Optional[str] = None
    chapter: Optional[str] = None
    heading: Optional[str] = None
    subheading: Optional[str] = None
    national: Optional[str] = None


class Alternative(BaseModel):
    national_code: str
    reasoning: str


class ClassificationResponse(BaseModel):
    request_id: str
    timestamp: str
    original_query: str
    processed_query: Optional[str] = None
    confidence: str = Field(
        description="HIGH | LOW | NO MATCH"
    )
    national_code: Optional[str] = Field(
        default=None,
        description="Recommended national commodity code"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the matched commodity code"
    )
    reasoning: str = Field(
        description="Explanation of the classification decision"
    )
    hierarchy_path: Optional[HierarchyPath] = None
    alternatives: Optional[list[Alternative]] = None
    exclusions_noted: Optional[str] = None
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    index_loaded: bool
    index_size: int
    timestamp: str


class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    results: list[dict]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    summary="Classify a product description",
    tags=["Classification"]
)
async def classify(request: ClassificationRequest):
    """
    Classify a product description into a national commodity code.

    - **HIGH confidence**: Single code returned with clear reasoning
    - **LOW confidence**: Up to 3 alternatives with reasoning
    - **NO MATCH**: Classification could not be reliably determined
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    request_id = request.request_id or str(uuid.uuid4())

    try:
        result = classifier.classify(
            query=request.description,
            request_id=request_id
        )

        # map hierarchy_path dict to model
        hierarchy_path = None
        if result.get("hierarchy_path"):
            hierarchy_path = HierarchyPath(**result["hierarchy_path"])

        # map alternatives list to model
        alternatives = None
        if result.get("alternatives"):
            alternatives = [
                Alternative(**a) for a in result["alternatives"]
                if isinstance(a, dict) and "national_code" in a
            ]

        return ClassificationResponse(
            request_id=result.get("request_id", request_id),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            original_query=result.get("original_query", request.description),
            processed_query=result.get("processed_query"),
            confidence=result.get("confidence", "LOW"),
            national_code=result.get("national_code"),
            description=result.get("description"),
            reasoning=result.get("reasoning", ""),
            hierarchy_path=hierarchy_path,
            alternatives=alternatives,
            exclusions_noted=result.get("exclusions_noted"),
            processing_time_ms=result.get("processing_time_ms", 0)
        )

    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Classification timed out — LLM took too long to respond. "
                "Consider increasing timeout in config/llm.yaml"
            )
        )
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal classification error — check server logs"
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"]
)
async def health():
    """Check if all system components are running correctly."""
    if not classifier:
        return HealthResponse(
            status="initializing",
            llm_available=False,
            index_loaded=False,
            index_size=0,
            timestamp=datetime.now().isoformat()
        )

    return HealthResponse(
        status="ok",
        llm_available=classifier.llm_client.ping(),
        index_loaded=classifier.retriever.is_loaded,
        index_size=classifier.retriever.index_size,
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/api/v1/reindex",
    summary="Trigger commodity code index rebuild",
    tags=["System"]
)
async def reindex(background_tasks: BackgroundTasks):
    """
    Trigger a full rebuild of the search indexes from the database.
    Runs in the background — returns immediately.
    Call this when commodity code data changes.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    background_tasks.add_task(classifier.rebuild_index)

    return {
        "status": "reindex started",
        "message": (
            "Index rebuild is running in the background. "
            "The system remains available during rebuild."
        )
    }


@app.get(
    "/api/v1/history",
    response_model=HistoryResponse,
    summary="Get classification history",
    tags=["History"]
)
async def get_history(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    confidence: Optional[str] = Query(
        default=None,
        description="Filter by confidence: HIGH | LOW | NO MATCH"
    ),
    national_code: Optional[str] = Query(
        default=None,
        description="Filter by specific national commodity code"
    )
):
    """Retrieve paginated classification history with optional filters."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    return classifier.history_logger.get_history(
        page=page,
        page_size=page_size,
        confidence_filter=confidence,
        national_code_filter=national_code
    )


@app.get(
    "/api/v1/history/{request_id}",
    summary="Get a specific classification by ID",
    tags=["History"]
)
async def get_classification(request_id: str):
    """Retrieve a single classification result by its request ID."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    result = classifier.history_logger.get_by_id(request_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Classification '{request_id}' not found"
        )
    return result
