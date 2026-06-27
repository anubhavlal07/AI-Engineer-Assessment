"""Central configuration loaded from environment / .env.

Single source of truth for models, paths, and retrieval tuning so every
component (ingestion, retrieval, generation, API, UI, eval) reads the same
values. No secrets are hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/config.py -> project root).
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Google Gemini ---
    # Read from GEMINI_API_KEY (preferred) or GOOGLE_API_KEY.
    gemini_api_key: str = ""
    google_api_key: str = ""

    # --- Models ---
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536  # gemini-embedding-001 supports 768 / 1536 / 3072

    # --- API auth ---
    api_key: str = "change-me-dev-key"

    # --- Retrieval tuning ---
    semantic_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5
    # Soft floor only (the reranker always keeps at least the top candidate).
    # 0.0 effectively defers the refusal decision to the grounded LLM prompt,
    # which is robust across document types; raise it to trim weak tail chunks.
    rerank_min_score: float = 0.0

    # --- Chunking ---
    chunk_tokens: int = 500
    chunk_overlap: int = 80

    # --- Generation ---
    llm_temperature: float = 0.0
    max_context_chunks: int = 5

    # --- Paths (relative to project root) ---
    data_raw_dir: str = "data/raw"
    data_index_dir: str = "data/index"

    # --- Re-ranker model (local cross-encoder) ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- UI ---
    ui_password: str = "demo"

    @property
    def resolved_api_key(self) -> str:
        """Gemini key, preferring GEMINI_API_KEY then GOOGLE_API_KEY."""
        return self.gemini_api_key or self.google_api_key

    @property
    def raw_path(self) -> Path:
        return ROOT_DIR / self.data_raw_dir

    @property
    def index_path(self) -> Path:
        return ROOT_DIR / self.data_index_dir

    @property
    def feedback_path(self) -> Path:
        return ROOT_DIR / "data" / "feedback.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Use this everywhere instead of constructing Settings()."""
    return Settings()
