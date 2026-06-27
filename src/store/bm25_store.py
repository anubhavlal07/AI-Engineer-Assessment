"""Keyword (lexical) store backed by BM25.

Complements semantic search by catching exact terms, IDs, and acronyms that
embeddings sometimes miss. Persisted as a pickle alongside the Chroma index.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import get_settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Store:
    """In-memory BM25 index with disk persistence.

    Holds parallel lists of chunk_ids and tokenized documents. Rebuilt from the
    full chunk set on each ingest (cheap for hundreds of docs).
    """

    def __init__(self) -> None:
        self._path = get_settings().index_path / "bm25.pkl"
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    # --- build / persist ---
    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        self._chunk_ids = chunk_ids
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as fh:
            pickle.dump({"chunk_ids": self._chunk_ids, "bm25": self._bm25}, fh)

    def load(self) -> bool:
        if not self._path.exists():
            return False
        with self._path.open("rb") as fh:
            data = pickle.load(fh)
        self._chunk_ids = data["chunk_ids"]
        self._bm25 = data["bm25"]
        return True

    # --- query ---
    def search(self, query: str, k: int) -> list[BM25Hit]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [BM25Hit(chunk_id=cid, score=float(s)) for cid, s in ranked[:k]]
