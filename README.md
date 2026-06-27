# Enterprise Knowledge Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** system that answers
natural-language questions over a collection of internal company documents
(HR policies, product docs, FAQs, compliance guidelines, …). It retrieves
relevant passages, generates a concise grounded answer, **cites the source
document and page**, and clearly says when an answer is not in the knowledge base.

Built for the Anthr-a-sync AI Engineer assignment.

---

## Highlights

- **Hybrid retrieval** — semantic (embeddings) **+** BM25 keyword search fused with
  Reciprocal Rank Fusion (RRF).
- **Cross-encoder re-ranking** for precision, with a relevance threshold that
  drives a graceful "not found" response.
- **Grounded generation** with mandatory inline citations and a strict
  anti-hallucination contract (temperature 0, JSON mode).
- **Conversation memory** with history-aware query rewriting for follow-ups.
- **Spec-compliant API** (`POST /ask` → `{answer, sources:[{document, page}], confidence}`).
- **Streamlit chat UI** with in-app document upload + management, sources panel,
  confidence badge, and feedback.
- **Evaluation harness** — golden set + retrieval metrics (Hit@k, MRR, Recall@k)
  + LLM-judged faithfulness/relevance + refusal accuracy.
- **Auth** (API key + UI password) and **Docker** deployment.

---

## Architecture

```
data/raw  ──►  Loader ──► Chunker ──► Embedder ──► Chroma (vectors) + BM25 (keywords)
                                                        │
User ─► (API | UI) ─► Query rewrite ─► Hybrid retrieve ─► Re-rank ─► LLM ─► answer + sources + confidence
```

Full diagram: [`docs/architecture.mmd`](docs/architecture.mmd) ·
Design doc: [`docs/system_design.md`](docs/system_design.md)

---

## Tech choices & rationale

| Concern | Choice | Why |
|---|---|---|
| LLM | Google `gemini-2.5-flash` (configurable) | Strong grounded QA, low cost, fast. |
| Embeddings | Google `gemini-embedding-001` (1536-d) | Task-type aware (doc vs query); 768/1536/3072 dims. |
| Vector DB | ChromaDB (persistent) | Zero-infra, metadata filtering; behind an interface. |
| Keyword | `rank_bm25` | Exact terms/IDs embeddings miss. |
| Fusion | Reciprocal Rank Fusion | Robust, scale-agnostic hybrid merge. |
| Re-ranker | `ms-marco-MiniLM-L-6-v2` cross-encoder (local) | Big precision win, no extra key. |
| API / UI | FastAPI + Streamlit | Matches spec; fast to demo. |
| Parsing | `pypdf`, `python-docx` | Per-page text → accurate page citations. |

No heavyweight RAG framework is used — orchestration is lean custom code with
each layer behind a small interface, for transparency and easy swapping (e.g. the
LLM/embedding provider is isolated to two wrapper modules).

---

## Setup

Requires **Python 3.11+** and a **Google Gemini API key**
(free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env        # then edit .env and set GEMINI_API_KEY + API_KEY
```

> The cross-encoder model (~80 MB) downloads automatically on first use.

---

## Usage

### 1. Add documents
Two ways:
- **From the UI (recommended):** launch the app (step 4) and use the sidebar
  **Upload & index** control — upload `.pdf` / `.docx` / `.md` / `.txt` files and
  they're indexed into the shared knowledge base on the spot. A **Manage
  documents** panel lets you see and delete what's indexed.
- **From the terminal:** drop files into `data/raw/` and run the ingest command below.

**Try it out-of-the-box** with the bundled sample documents (a synthetic HR policy
and customer policy that match the questions in `eval/golden_set.jsonl`):
```bash
cp sample_docs/* data/raw/      # Windows: copy sample_docs\* data\raw\
python -m scripts.ingest
```

### 2. Ingest (build the index — optional if you upload via the UI/API)
```bash
python -m scripts.ingest          # incremental
python -m scripts.ingest --force  # rebuild from scratch
```

### 3. Run the API
```bash
uvicorn src.api.main:app --reload
# Interactive docs at http://localhost:8000/docs
```

Example request:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your API_KEY>" \
  -d '{"question": "What is the refund policy?"}'
```
Response:
```json
{
  "answer": "Refunds are allowed within 30 days. [1]",
  "sources": [{"document": "Customer_Policy.pdf", "page": 5}],
  "confidence": 0.88
}
```

Upload documents via the API instead of the UI:
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: <your API_KEY>" \
  -F "files=@/path/to/HR_Policy.pdf" -F "files=@/path/to/FAQ.txt"
```

### 4. Run the chat UI
```bash
streamlit run src/ui/app.py
# Sign in with UI_PASSWORD (default: demo)
# Use the sidebar to upload documents and manage the knowledge base.
```

### 5. Evaluate
The bundled `eval/golden_set.jsonl` already matches the sample documents, so after
ingesting them (step 1) you can run the harness as-is. For your own corpus, edit
the golden set to match:
```bash
python -m eval.run_eval
# Writes eval/results/report.md and report.json
```

### Docker
```bash
docker compose up --build
# API → http://localhost:8000 · UI → http://localhost:8501
```

---

## API endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | – | Liveness + index readiness |
| POST | `/ask` | `X-API-Key` | Ask a question |
| POST | `/upload` | `X-API-Key` | Upload document(s) (multipart) and index them |
| POST | `/ingest?force=true` | `X-API-Key` | (Re)build the index from `data/raw/` |
| POST | `/feedback` | `X-API-Key` | Record thumbs-up/down + comment |

---

## Evaluation approach

- **Golden set** of representative questions tagged `factual` / `multi_doc` /
  `ambiguous` / `out_of_scope`.
- **Retrieval:** Hit@k, MRR, Recall@k against expected source documents.
- **Answer quality:** LLM-as-judge scores faithfulness (groundedness) and
  relevance; out-of-scope questions are scored on **correct refusal**.
- **Improvement story:** the design lets you compare semantic-only → +hybrid →
  +re-rank by toggling components, which is reflected in the metrics report.

---

## Design decisions

- **Page-bounded chunking** so every citation maps to a real page number.
- **Hybrid + re-rank** because pure semantic search misses exact identifiers and
  pure keyword search misses paraphrase; the cross-encoder then maximizes
  precision on the small candidate set.
- **Grounding-gated answering** — the reranker keeps the top-N candidates by
  rank and the strict grounded prompt decides whether they answer the question
  (robust across document types); retrieval scores inform confidence rather than
  acting as a brittle absolute cutoff. The system refuses rather than hallucinates.
- **Confidence is a transparent heuristic** (top rerank score + retrieval
  coverage + model groundedness flag), *not* a calibrated probability.

---

## Known limitations

- Citations are **page-level**, not span-level.
- Confidence is heuristic, not calibrated.
- DOCX/MD/TXT are treated as a single page (no native pagination).
- Conversation memory is in-process (lost on restart; not multi-replica safe).
- Scanned/image PDFs need OCR (not included).
- Quality depends on Gemini API availability and the quality of the source documents.

## Future improvements

- Span-level highlighting in citations; managed vector DB for scale.
- Redis-backed sessions for horizontal scaling; async ingestion queue.
- Embedding/answer caching; optional GPU or hosted re-ranker.
- Continuous evaluation dashboard fed by logged groundedness + user feedback.
- OAuth2/JWT per-user auth; OCR for scanned documents.

---

## Project structure

```
src/
  config.py              # pydantic settings
  ingestion/             # loader, chunker, pipeline
  embeddings/embedder.py # Gemini embeddings (batched, retried)
  store/                 # Chroma + BM25
  retrieval/             # hybrid (RRF) + cross-encoder reranker
  generation/            # prompts + answerer (citations, confidence)
  memory/conversation.py # session history + query rewrite
  rag_pipeline.py        # end-to-end orchestrator
  api/                   # FastAPI app, auth, schemas
  ui/app.py              # Streamlit chat
eval/                    # golden set, metrics, runner
tests/                   # pytest unit + API tests
scripts/                 # ingest CLI + sample-doc generator
sample_docs/             # bundled synthetic docs (match the golden set)
docs/                    # system design, diagram, demo script
```

## Assumptions

- You provide your own documents in `data/raw/` (no sample corpus is shipped).
- A Google Gemini API key is available (free tier is sufficient for the demo).
- Demo scale is single-node (hundreds of documents); scaling is discussed in the
  design doc.
