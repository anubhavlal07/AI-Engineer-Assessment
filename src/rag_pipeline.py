"""End-to-end RAG orchestrator.

Single entry point reused by the API, the UI, and the evaluation harness:

    rewrite query (memory) -> hybrid retrieve -> rerank -> generate -> remember

Components are constructed once and held, so the cross-encoder and Chroma client
are loaded a single time per process.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.embeddings.embedder import Embedder
from src.generation.answerer import Answer, Answerer
from src.ingestion.pipeline import load_chunk_corpus
from src.memory.conversation import ConversationMemory
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.store.bm25_store import BM25Store
from src.store.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: Answer
    retrieved: list[RetrievedChunk]   # post-rerank chunks (for UI/debug/eval)
    rewritten_query: str


class RAGPipeline:
    def __init__(self) -> None:
        self._embedder = Embedder()
        self._vector_store = VectorStore()
        self._bm25 = BM25Store()
        if not self._bm25.load():
            logger.warning("BM25 index not found. Run ingestion first.")
        self._corpus = load_chunk_corpus()
        self._retriever = HybridRetriever(
            self._vector_store, self._bm25, self._embedder, self._corpus
        )
        self._reranker = Reranker()
        self._answerer = Answerer()
        self._memory = ConversationMemory()

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def refresh(self) -> None:
        """Reload indexes after (re-)ingestion, in place.

        Only the parts that ingestion changes are rebuilt — BM25, the chunk
        corpus, and the retriever that wraps them. The Chroma vector store uses
        the same persistent client (already sees new chunks), and the expensive
        cross-encoder reranker / LLM clients are deliberately left untouched so
        callers (e.g. the Streamlit cached singleton) get a fast refresh.
        """
        self._bm25 = BM25Store()
        self._bm25.load()
        self._corpus = load_chunk_corpus()
        self._retriever = HybridRetriever(
            self._vector_store, self._bm25, self._embedder, self._corpus
        )

    def is_ready(self) -> bool:
        return self._vector_store.count() > 0

    def answer(
        self, question: str, session_id: str = "default", use_memory: bool = True
    ) -> PipelineResult:
        # 1. History-aware query rewrite.
        if use_memory:
            search_query = self._memory.rewrite_query(session_id, question)
        else:
            search_query = question
        if search_query != question:
            logger.info("Rewrote query -> %s", search_query)

        # 2. Hybrid retrieval + 3. rerank.
        candidates = self._retriever.retrieve(search_query)
        reranked = self._reranker.rerank(search_query, candidates)

        # 4. Generate grounded answer.
        history = self._memory.history_text(session_id) if use_memory else ""
        answer = self._answerer.answer(question, reranked, history=history)

        # 5. Remember the turn.
        if use_memory:
            self._memory.add_turn(session_id, question, answer.answer)

        return PipelineResult(
            answer=answer, retrieved=reranked, rewritten_query=search_query
        )


_PIPELINE: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Lazy process-wide singleton."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline()
    return _PIPELINE


def reset_pipeline() -> None:
    """Force re-init (e.g. after re-ingestion adds new chunks)."""
    global _PIPELINE
    _PIPELINE = None
