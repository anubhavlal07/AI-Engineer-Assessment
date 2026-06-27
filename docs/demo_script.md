# Demo Video Script (≤ 5 minutes)

A suggested running order for the screen recording.

## 0:00–0:30 — Intro & architecture (slide / diagram)
- "Enterprise Knowledge Assistant — a production-oriented RAG system."
- Show `docs/architecture.mmd` rendered. Name the path: ingest → hybrid retrieve
  → re-rank → grounded generate → cited answer.
- Mention stack: Google Gemini (LLM + embeddings), ChromaDB, BM25 + cross-encoder,
  FastAPI, Streamlit.

## 0:30–1:15 — Ingestion (upload from the UI)
- In the app sidebar, use **Upload & index** to upload a document live; show the
  spinner and the success summary (files indexed + chunk count).
- Open **Manage documents** to show the indexed list (and that a doc can be deleted).
- (Optional) Mention the CLI/`POST /upload` alternatives and the incremental
  manifest skip on re-ingest.

## 1:15–3:00 — Question answering (UI)
- `streamlit run src/ui/app.py`, sign in.
- Ask a factual question (e.g. leave policy). Show the answer, the **confidence
  badge**, and expand **Sources** (document + page) and **Retrieved passages**.
- Ask a **follow-up** ("what about ...?") to demonstrate conversation memory /
  query rewriting.
- Ask an **out-of-scope** question → show the graceful "could not find" response.
- Give a thumbs-up / thumbs-down to show feedback capture.

## 3:00–4:00 — API
- Show `GET /health`, then `POST /ask` via the FastAPI `/docs` page with the
  `X-API-Key` header. Highlight the spec-compliant JSON:
  `{ answer, sources:[{document, page}], confidence }`.

## 4:00–4:45 — Evaluation
- Run `python -m eval.run_eval`. Open `eval/results/report.md` and walk through
  retrieval metrics (Hit@k, MRR), faithfulness/relevance, and refusal accuracy.
- Mention the baseline → +hybrid → +rerank improvement story.

## 4:45–5:00 — Wrap
- Recap design choices and limitations (single-node demo, page-level citation,
  confidence is a heuristic). Point to README for full details.
