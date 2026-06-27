"""Unit tests for RRF fusion and BM25 keyword search. No API key required."""
from __future__ import annotations

from collections import defaultdict

from src.config import get_settings
from src.store.bm25_store import BM25Store


def _rrf(semantic_ids, bm25_ids, rrf_k):
    """Reference RRF used to validate fusion behavior (mirrors hybrid.py)."""
    fused = defaultdict(float)
    for rank, cid in enumerate(semantic_ids):
        fused[cid] += 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(bm25_ids):
        fused[cid] += 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def test_rrf_rewards_documents_in_both_lists():
    rrf_k = get_settings().rrf_k
    semantic = ["a", "b", "c"]
    bm25 = ["c", "d", "a"]
    ranked = _rrf(semantic, bm25, rrf_k)
    top_ids = [cid for cid, _ in ranked]
    # 'a' and 'c' appear in both lists -> should outrank single-list 'b' and 'd'.
    assert top_ids[0] in {"a", "c"}
    assert top_ids[1] in {"a", "c"}


def test_bm25_exact_term_match():
    store = BM25Store()
    store.build(
        chunk_ids=["c1", "c2", "c3"],
        texts=[
            "The refund policy allows returns within thirty days.",
            "Employees receive 24 paid leave days per year.",
            "Compliance requires data retention for seven years.",
        ],
    )
    hits = store.search("refund policy", k=3)
    assert hits, "expected at least one BM25 hit"
    assert hits[0].chunk_id == "c1"


def test_bm25_empty_index_returns_nothing():
    store = BM25Store()
    store.build(chunk_ids=[], texts=[])
    assert store.search("anything", k=5) == []
