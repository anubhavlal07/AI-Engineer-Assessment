"""Token-aware chunking.

Splits page segments into overlapping, token-bounded chunks. Splitting respects
paragraph boundaries first and never crosses a page boundary, so every chunk
maps cleanly to a single source page for citation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

import tiktoken

from src.config import get_settings
from src.ingestion.loader import PageSegment

# cl100k_base is a good general-purpose tokenizer for sizing chunks; it is used
# only to bound chunk length, independent of the embedding provider.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """An indexed unit of text with citation metadata."""

    chunk_id: str
    text: str
    source: str
    page: int
    doc_type: str
    heading: str
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _num_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _guess_heading(text: str) -> str:
    """Best-effort heading: first short, title-ish line of the segment."""
    for line in text.splitlines():
        line = line.strip()
        if 0 < len(line) <= 80 and not line.endswith("."):
            return line
    return ""


def _split_paragraphs(text: str) -> list[str]:
    # Split on blank lines; fall back to single newlines for dense text.
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _pack_paragraphs(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Greedily pack paragraphs into <= max_tokens windows.

    A paragraph longer than max_tokens is hard-split on token boundaries.
    """
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        ptokens = _num_tokens(para)
        if ptokens > max_tokens:
            # Flush current, then hard-split the oversized paragraph.
            if current:
                windows.append("\n\n".join(current))
                current, current_tokens = [], 0
            windows.extend(_hard_split(para, max_tokens))
            continue
        if current_tokens + ptokens > max_tokens and current:
            windows.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += ptokens

    if current:
        windows.append("\n\n".join(current))
    return windows


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _ENCODER.encode(text)
    return [
        _ENCODER.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _add_overlap(windows: list[str], overlap_tokens: int) -> list[str]:
    """Prepend a token-bounded tail of the previous window to each window."""
    if overlap_tokens <= 0 or len(windows) <= 1:
        return windows
    out = [windows[0]]
    for prev, curr in zip(windows, windows[1:]):
        prev_tokens = _ENCODER.encode(prev)
        tail = _ENCODER.decode(prev_tokens[-overlap_tokens:])
        out.append(f"{tail}\n\n{curr}")
    return out


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}::p{page}::c{index}::{digest}"


def chunk_segment(segment: PageSegment) -> list[Chunk]:
    settings = get_settings()
    heading = _guess_heading(segment.text)
    windows = _pack_paragraphs(_split_paragraphs(segment.text), settings.chunk_tokens)
    windows = _add_overlap(windows, settings.chunk_overlap)

    chunks: list[Chunk] = []
    for i, window in enumerate(windows):
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(segment.source, segment.page, i, window),
                text=window,
                source=segment.source,
                page=segment.page,
                doc_type=segment.doc_type,
                heading=heading,
                token_count=_num_tokens(window),
            )
        )
    return chunks


def chunk_segments(segments: list[PageSegment]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for seg in segments:
        chunks.extend(chunk_segment(seg))
    return chunks
