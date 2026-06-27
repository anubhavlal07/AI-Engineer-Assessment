"""FastAPI application exposing the Knowledge Assistant.

Endpoints:
    GET  /health    - liveness + index readiness
    POST /ask       - ask a question (auth)            -> spec-compliant response
    POST /ingest    - (re)build the index (auth)
    POST /feedback  - record thumbs up/down (auth)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.api.auth import require_api_key
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.api.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    HealthResponse,
    IngestResponse,
    RetrievedChunkModel,
    SourceModel,
)
from src.config import get_settings
from src.ingestion.pipeline import ingest
from src.rag_pipeline import get_pipeline, reset_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Production-oriented RAG API over internal documents.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        pipeline = get_pipeline()
        return HealthResponse(
            status="ok",
            index_ready=pipeline.is_ready(),
            total_chunks=pipeline._vector_store.count(),
        )
    except Exception as exc:  # noqa: BLE001
        return HealthResponse(status=f"degraded: {exc}", index_ready=False, total_chunks=0)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask(req: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=409,
            detail="Knowledge base is empty. Add documents to data/raw and call /ingest.",
        )
    result = pipeline.answer(req.question, req.session_id, req.use_memory)
    return AskResponse(
        answer=result.answer.answer,
        sources=[SourceModel(document=s.document, page=s.page) for s in result.answer.sources],
        confidence=result.answer.confidence,
        rewritten_query=result.rewritten_query,
        retrieved=[
            RetrievedChunkModel(
                source=c.source,
                page=c.page,
                heading=c.heading,
                score=round(c.score, 4),
                snippet=c.text[:300],
            )
            for c in result.retrieved
        ],
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest_endpoint(force: bool = False) -> IngestResponse:
    report = ingest(force=force)
    reset_pipeline()  # pick up the new index on next request
    return IngestResponse(**report.__dict__)


@app.post("/upload", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def upload(files: list[UploadFile] = File(...)) -> IngestResponse:
    raw_dir = get_settings().raw_path
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        content = await f.read()
        if not content:
            continue
        (raw_dir / Path(f.filename).name).write_bytes(content)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="No valid files were uploaded.")

    report = ingest()
    reset_pipeline()  # pick up the new index on next request
    return IngestResponse(**report.__dict__)


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def feedback(req: FeedbackRequest) -> dict:
    path = get_settings().feedback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(req.model_dump()) + "\n")
    return {"status": "recorded"}
