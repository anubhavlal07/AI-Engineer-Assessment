"""Answer generation from reranked context.

Builds numbered context blocks, calls the LLM in JSON mode at low temperature,
parses the grounded answer + the sources actually used, and computes a
transparent confidence heuristic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from src.store.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "I could not find this information in the knowledge base."


@dataclass
class Source:
    document: str
    page: int


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    confidence: float


class Answerer:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self._client = genai.Client(api_key=settings.resolved_api_key)
        self._settings = settings

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    def _call_llm(self, user_prompt: str) -> dict:
        response = self._client.models.generate_content(
            model=self._settings.llm_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self._settings.llm_temperature,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)

    def _confidence(
        self, chunks: list[RetrievedChunk], used: list[int], grounded: bool
    ) -> float:
        """Transparent heuristic (NOT a calibrated probability).

        Leans primarily on signals that are robust across document types: the
        model's self-reported groundedness and whether it actually cited
        sources. The absolute reranker score is only a soft bonus, because it
        is miscalibrated for terse text (e.g. résumés/tables score low even when
        correct) and must not, by itself, drag a well-grounded, cited answer
        down to near-zero confidence.
        """
        if not chunks:
            return 0.0
        top_score = max(c.score for c in chunks)
        cited = 1.0 if used else 0.0
        raw = 0.45 * (1.0 if grounded else 0.0) + 0.35 * cited + 0.20 * top_score
        return round(max(0.0, min(1.0, raw)), 2)

    def answer(
        self, question: str, chunks: list[RetrievedChunk], history: str = ""
    ) -> Answer:
        # No relevant context -> deterministic "unavailable" response.
        if not chunks:
            return Answer(answer=UNAVAILABLE_MESSAGE, sources=[], confidence=0.0)

        blocks = [
            build_context_block(i + 1, c.source, c.page, c.text)
            for i, c in enumerate(chunks)
        ]
        user_prompt = build_user_prompt(question, blocks, history)

        try:
            parsed = self._call_llm(user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed: %s", exc)
            raise

        answer_text = (parsed.get("answer") or "").strip() or UNAVAILABLE_MESSAGE
        used = parsed.get("used_sources") or []
        used = [int(n) for n in used if isinstance(n, (int, float, str)) and str(n).isdigit()]
        grounded = bool(parsed.get("grounded", False))

        # Map used block numbers back to citation sources (deduped, order-preserving).
        sources: list[Source] = []
        seen: set[tuple[str, int]] = set()
        for n in used:
            if 1 <= n <= len(chunks):
                c = chunks[n - 1]
                key = (c.source, c.page)
                if key not in seen:
                    seen.add(key)
                    sources.append(Source(document=c.source, page=c.page))

        # If the model returned the unavailable message, no sources/zero confidence.
        if answer_text == UNAVAILABLE_MESSAGE:
            return Answer(answer=answer_text, sources=[], confidence=0.0)

        confidence = self._confidence(chunks, used, grounded)
        return Answer(answer=answer_text, sources=sources, confidence=confidence)
