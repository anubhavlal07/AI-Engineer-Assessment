"""Evaluation metrics for retrieval and answer quality.

Retrieval metrics are computed from the documents that appear in the retrieved
chunks vs. the expected source documents. Answer-quality metrics
(faithfulness, relevance) use an LLM-as-judge.
"""
from __future__ import annotations

import json

from google import genai
from google.genai import types

from src.config import get_settings


# ----------------------------------------------------------------- retrieval
def hit_at_k(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    """1.0 if any expected document appears in the retrieved set, else 0.0."""
    if not expected_docs:
        return 1.0  # nothing to retrieve (out-of-scope) -> trivially satisfied
    return 1.0 if set(retrieved_docs) & set(expected_docs) else 0.0


def mrr(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    """Reciprocal rank of the first expected document in the retrieved list."""
    if not expected_docs:
        return 1.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected_docs:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    if not expected_docs:
        return 1.0
    found = set(retrieved_docs) & set(expected_docs)
    return len(found) / len(set(expected_docs))


# ----------------------------------------------------------------- answer (LLM judge)
_JUDGE_SYSTEM = (
    "You are a strict evaluator of a RAG assistant. Given a question, the "
    "retrieved context, and the assistant's answer, score two things from 0 to 1:\n"
    "- faithfulness: is every claim in the answer supported by the context? "
    "(1 = fully grounded, 0 = hallucinated)\n"
    "- relevance: does the answer actually address the question?\n"
    "Return ONLY JSON: {\"faithfulness\": <0-1>, \"relevance\": <0-1>}"
)


class AnswerJudge:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = genai.Client(api_key=settings.resolved_api_key)
        self._model = settings.llm_model

    def judge(self, question: str, context: str, answer: str) -> dict:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
            config=types.GenerateContentConfig(
                system_instruction=_JUDGE_SYSTEM,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(resp.text)
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "relevance": float(data.get("relevance", 0.0)),
        }


def refusal_correct(answer: str, expected_type: str) -> float | None:
    """For out-of-scope items, reward correct refusal. Returns None if N/A."""
    if expected_type != "out_of_scope":
        return None
    refused = "could not find" in answer.lower()
    return 1.0 if refused else 0.0
