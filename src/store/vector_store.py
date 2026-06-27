"""Persistent semantic store backed by ChromaDB.

Wraps a single Chroma collection. Stores chunk text + citation metadata and
returns cosine-similarity-ranked results. The implementation is hidden behind a
small interface so it can be swapped for a managed vector DB later.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src.config import get_settings
from src.ingestion.chunker import Chunk

_COLLECTION = "knowledge_base"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    doc_type: str
    heading: str
    score: float  # higher = more relevant


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        settings.index_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.index_path / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page,
                    "doc_type": c.doc_type,
                    "heading": c.heading,
                }
                for c in chunks
            ],
        )

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def query(
        self, embedding: list[float], k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        retrieved = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    source=meta.get("source", "unknown"),
                    page=int(meta.get("page", 1)),
                    doc_type=meta.get("doc_type", ""),
                    heading=meta.get("heading", ""),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return retrieved
