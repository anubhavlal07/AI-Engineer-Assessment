"""Pydantic request/response models. The /ask response matches the assignment
spec exactly: {answer, sources:[{document, page}], confidence}."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    session_id: str = Field("default", description="Conversation session id")
    use_memory: bool = Field(True, description="Use conversation history for follow-ups")


class SourceModel(BaseModel):
    document: str
    page: int


class RetrievedChunkModel(BaseModel):
    source: str
    page: int
    heading: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    confidence: float
    rewritten_query: str | None = None
    retrieved: list[RetrievedChunkModel] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files_processed: int
    files_skipped: int
    chunks_indexed: int
    total_chunks: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool
    comment: str = ""
    session_id: str = "default"


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    total_chunks: int
