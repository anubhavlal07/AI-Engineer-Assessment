"""Conversation memory + history-aware query rewriting.

Keeps a short per-session chat history (in-memory; swappable for Redis). Before
retrieval, follow-up questions like "what about for contractors?" are condensed
into standalone queries so retrieval has full context.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque

from google import genai
from google.genai import types

from src.config import get_settings

logger = logging.getLogger(__name__)

_MAX_TURNS = 6  # keep the last N (user, assistant) exchanges per session

_REWRITE_SYSTEM = (
    "Given a conversation history and a follow-up question, rewrite the follow-up "
    "as a standalone question that can be understood without the history. If it is "
    "already standalone, return it unchanged. Return ONLY the rewritten question."
)


class ConversationMemory:
    def __init__(self) -> None:
        self._sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_TURNS))
        settings = get_settings()
        key = settings.resolved_api_key
        self._client = genai.Client(api_key=key) if key else None
        self._model = settings.llm_model

    def history_text(self, session_id: str) -> str:
        turns = self._sessions.get(session_id)
        if not turns:
            return ""
        lines = []
        for user_msg, assistant_msg in turns:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
        return "\n".join(lines)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        self._sessions[session_id].append((question, answer))

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def rewrite_query(self, session_id: str, question: str) -> str:
        """Condense a follow-up into a standalone query. Falls back to the
        original question on any error or when there is no history."""
        history = self.history_text(session_id)
        if not history or self._client is None:
            return question
        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=f"History:\n{history}\n\nFollow-up: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=_REWRITE_SYSTEM,
                    temperature=0.0,
                ),
            )
            rewritten = (resp.text or "").strip()
            return rewritten or question
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed, using original: %s", exc)
            return question
