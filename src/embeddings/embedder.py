"""Google Gemini embeddings wrapper.

Batches requests, retries on transient errors with exponential backoff, and
exposes a small interface used by both ingestion and query-time retrieval.

Uses task-type asymmetry for better retrieval: documents are embedded with
``RETRIEVAL_DOCUMENT`` and queries with ``RETRIEVAL_QUERY``. Vectors are
L2-normalized, which Google recommends when using a reduced output
dimensionality (768/1536 instead of the default 3072).
"""
from __future__ import annotations

import logging
import math

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # Gemini embed_content accepts a list of inputs per call.


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.resolved_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self._client = genai.Client(api_key=settings.resolved_api_key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dim,
            ),
        )
        return [_normalize(list(e.values)) for e in response.embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (RETRIEVAL_DOCUMENT) in batches."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
            logger.info(
                "Embedded %d/%d texts", min(start + _BATCH_SIZE, len(texts)), len(texts)
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]
