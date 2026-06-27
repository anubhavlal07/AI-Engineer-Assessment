"""CLI entry point for ingestion.

Usage:
    python -m scripts.ingest            # incremental (skip unchanged files)
    python -m scripts.ingest --force    # rebuild the whole index
"""
from __future__ import annotations

import argparse
import logging

from src.config import get_settings
from src.ingestion.pipeline import ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the index.")
    parser.add_argument("--force", action="store_true", help="Rebuild the entire index.")
    args = parser.parse_args()

    raw_dir = get_settings().raw_path
    print(f"Ingesting documents from: {raw_dir}")
    report = ingest(force=args.force)
    print(
        f"Done. processed={report.files_processed} skipped={report.files_skipped} "
        f"new_chunks={report.chunks_indexed} total_chunks={report.total_chunks}"
    )
    if report.total_chunks == 0:
        print(
            "\nNo chunks indexed. Add .pdf/.docx/.md/.txt files to "
            f"{raw_dir} and run again."
        )


if __name__ == "__main__":
    main()
