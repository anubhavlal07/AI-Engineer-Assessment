"""Cross-encoder re-ranking.

A bi-encoder (embeddings) is fast but coarse. A cross-encoder jointly reads the
(query, chunk) pair and scores true relevance far more precisely. We rerank the
fused candidate set and keep the top-n above a minimum relevance threshold.
Candidates below the threshold are dropped, which drives the
"information unavailable" path when nothing is relevant.

The model is loaded lazily and cached so the API/UI pay the cost once.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.config import get_settings
from src.store.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    model_name = get_settings().reranker_model
    logger.info("Loading cross-encoder re-ranker: %s", model_name)
    return CrossEncoder(model_name)


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self) -> None:
        self._settings = get_settings()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = _get_model()
        pairs = [(query, c.text) for c in candidates]
        raw_scores = model.predict(pairs)

        scored = []
        for cand, raw in zip(candidates, raw_scores):
            cand.score = _sigmoid(float(raw))  # normalize logits to 0..1
            scored.append(cand)

        scored.sort(key=lambda c: c.score, reverse=True)

        # Keep the top-N candidates by RANK and let the grounded LLM prompt
        # decide whether they actually answer the question. Absolute
        # cross-encoder scores vary enormously by document type (terse résumés
        # and tables score far lower than prose, even when correct), so a fixed
        # absolute floor as a hard refusal gate causes false refusals. The
        # optional min-score floor only trims weak tail chunks and can never
        # empty a non-empty candidate set (we always keep at least the best one).
        top_n = scored[: self._settings.rerank_top_n]
        floor = self._settings.rerank_min_score
        kept = [c for c in top_n if c.score >= floor]
        return kept or top_n[:1]
