"""Run the golden-set evaluation through the RAG pipeline and write a report.

Usage:
    python -m eval.run_eval

Produces eval/results/report.md and eval/results/report.json with per-question
and aggregate retrieval + answer-quality metrics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean

from eval.metrics import (
    AnswerJudge,
    hit_at_k,
    mrr,
    recall_at_k,
    refusal_correct,
)
from src.config import ROOT_DIR
from src.rag_pipeline import get_pipeline

logging.basicConfig(level=logging.WARNING)

GOLDEN = ROOT_DIR / "eval" / "golden_set.jsonl"
RESULTS_DIR = ROOT_DIR / "eval" / "results"


def _load_golden() -> list[dict]:
    items = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def main() -> None:
    pipeline = get_pipeline()
    if not pipeline.is_ready():
        print("Index is empty. Run `python -m scripts.ingest` first.")
        return

    judge = AnswerJudge()
    items = _load_golden()
    rows: list[dict] = []

    for item in items:
        # Each eval question is independent -> no conversation memory.
        result = pipeline.answer(item["question"], session_id="eval", use_memory=False)
        retrieved_docs = [c.source for c in result.retrieved]
        context = "\n\n".join(c.text for c in result.retrieved)
        answer = result.answer.answer

        row = {
            "question": item["question"],
            "type": item.get("type", "factual"),
            "answer": answer,
            "confidence": result.answer.confidence,
            "retrieved_docs": retrieved_docs,
            "hit@k": hit_at_k(retrieved_docs, item.get("expected_sources", [])),
            "mrr": mrr(retrieved_docs, item.get("expected_sources", [])),
            "recall@k": recall_at_k(retrieved_docs, item.get("expected_sources", [])),
        }

        refusal = refusal_correct(answer, item.get("type", ""))
        if refusal is not None:
            row["refusal_correct"] = refusal
        else:
            scores = judge.judge(item["question"], context, answer)
            row.update(scores)
        rows.append(row)

    _write_report(rows)


def _agg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return round(mean(vals), 3) if vals else 0.0


def _write_report(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "n": len(rows),
        "hit@k": _agg(rows, "hit@k"),
        "mrr": _agg(rows, "mrr"),
        "recall@k": _agg(rows, "recall@k"),
        "faithfulness": _agg(rows, "faithfulness"),
        "relevance": _agg(rows, "relevance"),
        "refusal_correct": _agg(rows, "refusal_correct"),
    }

    (RESULTS_DIR / "report.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = ["# Evaluation Report", "", "## Aggregate metrics", ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per-question results", ""]
    lines.append("| Type | Question | Hit@k | MRR | Faith | Rel | Refusal | Conf |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['type']} | {r['question'][:40]} | {r.get('hit@k','-')} | "
            f"{round(r.get('mrr',0),2)} | {r.get('faithfulness','-')} | "
            f"{r.get('relevance','-')} | {r.get('refusal_correct','-')} | {r['confidence']} |"
        )
    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote eval/results/report.md and report.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
