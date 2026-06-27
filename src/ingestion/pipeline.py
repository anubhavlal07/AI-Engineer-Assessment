"""Ingestion orchestrator: load -> chunk -> embed -> index.

Builds the semantic (Chroma) and lexical (BM25) indexes from documents in the
raw directory and writes a ``chunks.jsonl`` corpus + a content-hash manifest.
Idempotent: unchanged files are skipped on re-ingest; changed/removed files are
re-indexed so citations stay correct.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunk, chunk_segments
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_file
from src.store.bm25_store import BM25Store
from src.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files_processed: int
    files_skipped: int
    chunks_indexed: int
    total_chunks: int


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path() -> Path:
    return get_settings().index_path / "manifest.json"


def _chunks_path() -> Path:
    return get_settings().index_path / "chunks.jsonl"


def _load_manifest() -> dict[str, str]:
    path = _manifest_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_chunk_corpus() -> dict[str, Chunk]:
    """Load all persisted chunks keyed by chunk_id (used by retrieval/rerank)."""
    path = _chunks_path()
    corpus: dict[str, Chunk] = {}
    if not path.exists():
        return corpus
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            corpus[data["chunk_id"]] = Chunk(**data)
    return corpus


def _write_chunk_corpus(chunks: list[Chunk]) -> None:
    with _chunks_path().open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict()) + "\n")


def ingest(force: bool = False) -> IngestReport:
    """Run ingestion over the raw directory.

    Args:
        force: re-index every file even if unchanged.
    """
    settings = get_settings()
    settings.index_path.mkdir(parents=True, exist_ok=True)
    raw_dir = settings.raw_path

    vector_store = VectorStore()
    embedder = Embedder()

    manifest = {} if force else _load_manifest()
    existing_corpus = {} if force else load_chunk_corpus()
    if force:
        vector_store.reset()

    files = [
        p for p in sorted(raw_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    current_names = {p.name for p in files}

    # Drop chunks belonging to files that were removed from the raw dir.
    for removed in set(_files_in_manifest(manifest)) - current_names:
        logger.info("Removing deleted file from index: %s", removed)
        vector_store.delete_by_source(removed)
        existing_corpus = {
            cid: c for cid, c in existing_corpus.items() if c.source != removed
        }
        manifest.pop(removed, None)

    new_manifest = dict(manifest)
    processed, skipped = 0, 0
    chunks_added = 0
    corpus = dict(existing_corpus)

    for path in files:
        digest = _file_hash(path)
        if manifest.get(path.name) == digest:
            skipped += 1
            continue

        # Re-index this file: clear its old chunks first.
        vector_store.delete_by_source(path.name)
        corpus = {cid: c for cid, c in corpus.items() if c.source != path.name}

        segments = load_file(path)
        chunks = chunk_segments(segments)
        if chunks:
            embeddings = embedder.embed_texts([c.text for c in chunks])
            vector_store.add(chunks, embeddings)
            for c in chunks:
                corpus[c.chunk_id] = c
            chunks_added += len(chunks)
        new_manifest[path.name] = digest
        processed += 1
        logger.info("Indexed %s -> %d chunks", path.name, len(chunks))

    # Persist corpus + rebuild BM25 over the full set + manifest.
    all_chunks = list(corpus.values())
    _write_chunk_corpus(all_chunks)

    bm25 = BM25Store()
    bm25.build([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])
    bm25.save()

    _save_manifest(new_manifest)

    return IngestReport(
        files_processed=processed,
        files_skipped=skipped,
        chunks_indexed=chunks_added,
        total_chunks=len(all_chunks),
    )


def _files_in_manifest(manifest: dict[str, str]) -> list[str]:
    return list(manifest.keys())
