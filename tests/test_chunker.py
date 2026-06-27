"""Unit tests for token-aware chunking. No API key required."""
from __future__ import annotations

from src.config import get_settings
from src.ingestion.chunker import chunk_segment, _num_tokens
from src.ingestion.loader import PageSegment


def test_short_segment_single_chunk():
    seg = PageSegment(text="The leave policy grants 24 paid days.", source="hr.pdf", page=3, doc_type="pdf")
    chunks = chunk_segment(seg)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.source == "hr.pdf"
    assert c.page == 3
    assert c.doc_type == "pdf"
    assert "24 paid days" in c.text


def test_long_segment_splits_within_token_budget():
    settings = get_settings()
    para = ("word " * 400).strip()
    text = "\n\n".join([para, para, para])  # ~1200 words, well over the budget
    seg = PageSegment(text=text, source="big.txt", page=1, doc_type="txt")
    chunks = chunk_segment(seg)
    assert len(chunks) > 1
    # Each chunk should respect the token budget (allowing overlap headroom).
    budget = settings.chunk_tokens + settings.chunk_overlap + 50
    assert all(_num_tokens(c.text) <= budget for c in chunks)


def test_chunk_ids_are_unique():
    seg = PageSegment(text="\n\n".join(f"Paragraph number {i} content." for i in range(50)),
                      source="doc.md", page=1, doc_type="markdown")
    chunks = chunk_segment(seg)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_page_preserved_for_citation():
    seg = PageSegment(text="Refunds allowed within 30 days.", source="cust.pdf", page=5, doc_type="pdf")
    chunks = chunk_segment(seg)
    assert all(c.page == 5 for c in chunks)
