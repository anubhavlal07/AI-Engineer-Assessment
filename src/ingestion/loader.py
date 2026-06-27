"""Document loading and text extraction.

Dispatches by file extension and returns per-page text segments so that page
numbers survive into the index and can be cited. Each loader yields
``PageSegment`` records. Unreadable files are logged and skipped rather than
crashing the whole ingestion run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class PageSegment:
    """A unit of extracted text tied to a source document and page."""

    text: str
    source: str          # filename, e.g. "HR_Policy.pdf"
    page: int            # 1-based page number (1 for non-paginated formats)
    doc_type: str        # file extension without dot, e.g. "pdf"


def _load_pdf(path: Path) -> list[PageSegment]:
    from pypdf import PdfReader

    segments: list[PageSegment] = []
    reader = PdfReader(str(path))
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(
                PageSegment(text=text, source=path.name, page=idx, doc_type="pdf")
            )
    return segments


def _load_docx(path: Path) -> list[PageSegment]:
    import docx  # python-docx

    document = docx.Document(str(path))
    # DOCX has no reliable page boundaries; treat the whole doc as page 1.
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [PageSegment(text=text, source=path.name, page=1, doc_type="docx")]


def _load_text(path: Path, doc_type: str) -> list[PageSegment]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [PageSegment(text=text, source=path.name, page=1, doc_type=doc_type)]


def load_file(path: Path) -> list[PageSegment]:
    """Load a single file into page segments. Returns [] on failure."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return _load_pdf(path)
        if ext == ".docx":
            return _load_docx(path)
        if ext in {".md", ".markdown"}:
            return _load_text(path, "markdown")
        if ext == ".txt":
            return _load_text(path, "txt")
        logger.warning("Unsupported file type, skipping: %s", path.name)
        return []
    except Exception as exc:  # noqa: BLE001 - keep ingestion resilient
        logger.error("Failed to load %s: %s", path.name, exc)
        return []


def load_directory(directory: Path) -> list[PageSegment]:
    """Load every supported file in a directory (non-recursive)."""
    segments: list[PageSegment] = []
    files = [
        p for p in sorted(directory.glob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        logger.warning("No supported documents found in %s", directory)
    for path in files:
        loaded = load_file(path)
        logger.info("Loaded %d page segment(s) from %s", len(loaded), path.name)
        segments.extend(loaded)
    return segments
