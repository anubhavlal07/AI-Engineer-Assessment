# System Design — Enterprise Knowledge Assistant

## 1. High-level architecture

The system is a Retrieval-Augmented Generation (RAG) application with a clear
split between an **offline ingestion** path and an **online query** path,
exposed through both an API and a chat UI.

```
data/raw  ──►  Loader ──► Chunker ──► Embedder ──► Chroma (vectors) + BM25 (keywords)
                                                        │
User ─► (API | UI) ─► Query rewrite ─► Hybrid retrieve ─► Re-rank ─► LLM ─► answer+sources+confidence
```

See `docs/architecture.mmd` for the full diagram.

## 2. Data flow

1. **Ingest.** Documents in `data/raw/` are parsed per page (PDF/DOCX/MD/TXT),
   split into token-bounded chunks that never cross a page boundary (so each
   chunk cites exactly one page), embedded with Google `gemini-embedding-001`
   (documents use the `RETRIEVAL_DOCUMENT` task type), and written to a persistent
   Chroma collection. A parallel BM25 keyword index
   and a `chunks.jsonl` corpus + content-hash manifest are saved. Re-ingestion is
   incremental: unchanged files (by SHA-256) are skipped; changed/removed files
   are re-indexed.
2. **Query.** A question (plus optional conversation history) is condensed into a
   standalone query. Hybrid retrieval runs semantic top-k and BM25 top-k and
   fuses them with Reciprocal Rank Fusion (RRF). A cross-encoder re-ranks the
   fused candidates and keeps the top-n above a relevance threshold.
3. **Generate.** The top chunks become numbered context blocks. The LLM
   (`gemini-2.5-flash`) answers in JSON mode at temperature 0 under a strict grounding
   contract, returning the answer, the context blocks actually used, and a
   self-reported groundedness flag. The service maps used blocks to
   `{document, page}` citations and computes a confidence heuristic.
4. **Respond + learn.** The response (`answer`, `sources`, `confidence`) is
   returned; the turn is stored in conversation memory; user thumbs feedback is
   appended to `feedback.jsonl`.

## 3. Component explanation

| Component | Responsibility | Key file |
|---|---|---|
| Loader | Per-page text extraction, resilient to bad files | `src/ingestion/loader.py` |
| Chunker | Token-aware, page-bounded chunking with overlap | `src/ingestion/chunker.py` |
| Pipeline | Orchestrate ingest; idempotent manifest | `src/ingestion/pipeline.py` |
| Embedder | Batched Gemini embeddings (task-typed) + retry | `src/embeddings/embedder.py` |
| VectorStore | Chroma persistence + cosine query | `src/store/vector_store.py` |
| BM25Store | Lexical index + persistence | `src/store/bm25_store.py` |
| HybridRetriever | Semantic ∪ BM25 fused via RRF | `src/retrieval/hybrid.py` |
| Reranker | Cross-encoder precision re-ranking + threshold | `src/retrieval/reranker.py` |
| Answerer | Grounded generation, citations, confidence | `src/generation/answerer.py` |
| ConversationMemory | Session history + query rewrite | `src/memory/conversation.py` |
| RAGPipeline | End-to-end orchestrator (shared singleton) | `src/rag_pipeline.py` |
| API / UI | FastAPI endpoints + Streamlit chat | `src/api/`, `src/ui/` |

## 4. Hallucination & ambiguity control

- Strict system prompt: answer only from context, cite every claim, emit a fixed
  "could not find this information" message when context is insufficient.
- Temperature 0 + JSON-mode structured output.
- Relevance threshold after re-ranking → if nothing clears it, the answerer
  short-circuits to the unavailable response with zero confidence.
- Ambiguous queries: the model is instructed to state its interpretation or ask a
  brief clarifying question rather than guess.

## 5. Scalability considerations

- **Vector store:** Chroma is embedded for the demo. The `VectorStore` interface
  isolates it so it can be swapped for a managed/distributed store
  (Pinecone/Weaviate/pgvector) without touching retrieval logic.
- **Stateless API:** `/ask` is stateless apart from in-memory conversation
  history; moving sessions to Redis makes the API horizontally scalable behind a
  load balancer.
- **Ingestion at scale:** the current synchronous pipeline can be moved to a
  worker queue (Celery/RQ) for large corpora; embeddings are already batched and
  retried, and the manifest enables incremental indexing.
- **Cost/latency:** embeddings can be cached; frequent answers can be cached by
  normalized query; the re-ranker can run on GPU or be replaced by a hosted
  rerank API. Model choice (`gemini-2.5-flash` vs `gemini-2.5-pro`) is a config switch.
- **Observability (future):** per-stage latency, retrieval hit-rate, and
  groundedness can be logged to drive continuous evaluation.
