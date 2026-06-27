"""Hybrid retrieval: semantic + BM25 fused via Reciprocal Rank Fusion (RRF).

Semantic search captures meaning; BM25 captures exact lexical matches. RRF
merges the two ranked lists without needing comparable score scales:

    rrf_score(d) = sum over lists of 1 / (rrf_k + rank(d))
"""
from __future__ import annotations

from collections import defaultdict

from src.config import get_settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk
from src.store.bm25_store import BM25Store
from src.store.vector_store import RetrievedChunk, VectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        corpus: dict[str, Chunk],
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_store
        self._embedder = embedder
        self._corpus = corpus
        self._settings = get_settings()

    def _to_retrieved(self, chunk: Chunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            page=chunk.page,
            doc_type=chunk.doc_type,
            heading=chunk.heading,
            score=score,
        )

    def retrieve(self, query: str, where: dict | None = None) -> list[RetrievedChunk]:
        s = self._settings

        # 1. Semantic candidates (ranked).
        embedding = self._embedder.embed_query(query)
        semantic = self._vs.query(embedding, k=s.semantic_top_k, where=where)
        semantic_ranks = {rc.chunk_id: i for i, rc in enumerate(semantic)}
        by_id = {rc.chunk_id: rc for rc in semantic}

        # 2. Lexical candidates (ranked).
        bm25_hits = self._bm25.search(query, k=s.bm25_top_k)
        bm25_ranks = {hit.chunk_id: i for i, hit in enumerate(bm25_hits)}

        # 3. RRF fusion over the union of candidate ids.
        fused: dict[str, float] = defaultdict(float)
        for cid, rank in semantic_ranks.items():
            fused[cid] += 1.0 / (s.rrf_k + rank)
        for cid, rank in bm25_ranks.items():
            fused[cid] += 1.0 / (s.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for cid, rrf_score in ordered:
            if cid in by_id:
                rc = by_id[cid]
                rc.score = rrf_score
                results.append(rc)
            elif cid in self._corpus:
                results.append(self._to_retrieved(self._corpus[cid], rrf_score))
        return results
