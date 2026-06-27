"""API smoke tests. No Gemini key required — they exercise routing and auth,
not the LLM."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.config import get_settings

client = TestClient(app)


def test_health_returns_ok_shape():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "index_ready" in body
    assert "total_chunks" in body


def test_ask_requires_api_key():
    # Default config has a non-empty api_key, so a missing header -> 401.
    settings = get_settings()
    if not settings.api_key:
        return  # auth disabled in this env; nothing to assert
    resp = client.post("/ask", json={"question": "What is the leave policy?"})
    assert resp.status_code == 401


def test_ask_rejects_empty_question():
    settings = get_settings()
    resp = client.post(
        "/ask",
        json={"question": ""},
        headers={"X-API-Key": settings.api_key},
    )
    # Pydantic validation (min_length=1) -> 422 before any LLM work.
    assert resp.status_code == 422
