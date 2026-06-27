"""Streamlit chat UI for the Enterprise Knowledge Assistant.

Talks to the RAG pipeline in-process. Provides a password gate, chat with
conversation memory, an expandable Sources panel, a confidence badge, and
thumbs-up/down feedback.

Run:  streamlit run src/ui/app.py
"""
from __future__ import annotations

import json

import streamlit as st

from src.config import get_settings
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.ingestion.pipeline import delete_document, ingest, list_indexed_documents
from src.rag_pipeline import get_pipeline

st.set_page_config(page_title="Enterprise Knowledge Assistant")
settings = get_settings()

MAX_UPLOAD_MB = 25  # soft per-file cap; Streamlit's server.maxUploadSize still applies
UPLOAD_TYPES = ["pdf", "docx", "md", "markdown", "txt"]


# ---------------------------------------------------------------- auth gate
def _check_password() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("Enterprise Knowledge Assistant")
    pwd = st.text_input("Enter access password", type="password")
    if st.button("Sign in"):
        if pwd == settings.ui_password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()


# ---------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading models and index...")
def _pipeline():
    return get_pipeline()


pipeline = _pipeline()

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts: role, content, meta
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "ui-session"

st.title("Enterprise Knowledge Assistant")

with st.sidebar:
    st.subheader("Knowledge base")
    st.write(f"Indexed chunks: **{pipeline._vector_store.count()}**")

    # --- Upload & index ---
    uploaded = st.file_uploader(
        "Upload documents",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="PDF, DOCX, MD or TXT. Files are added to the shared knowledge base.",
    )
    if st.button("Upload & index", disabled=not uploaded):
        saved, rejected = [], []
        for f in uploaded:
            data = f.getvalue()
            if not data:
                rejected.append(f"{f.name} (empty)")
                continue
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                rejected.append(f"{f.name} (> {MAX_UPLOAD_MB} MB)")
                continue
            (settings.raw_path / f.name).write_bytes(data)
            saved.append(f.name)

        for msg in rejected:
            st.warning(f"Skipped {msg}")

        if saved:
            with st.spinner(f"Indexing {len(saved)} file(s)…"):
                report = ingest()
                pipeline.refresh()
            st.success(
                f"Indexed {report.files_processed} file(s) "
                f"({report.files_skipped} unchanged) · "
                f"+{report.chunks_indexed} chunks · "
                f"{report.total_chunks} total."
            )
            st.rerun()

    # --- Manage indexed documents ---
    docs = list_indexed_documents()
    with st.expander(f"Manage documents ({len(docs)})", expanded=not docs):
        if not docs:
            st.caption("No documents indexed yet. Upload some above.")
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{doc['source']}**  \n<small>{doc['chunks']} chunks</small>",
                        unsafe_allow_html=True)
            if c2.button("Delete", key=f"del-{doc['source']}", help="Delete from knowledge base"):
                with st.spinner(f"Removing {doc['source']}…"):
                    delete_document(doc["source"])
                    pipeline.refresh()
                st.rerun()

    st.divider()
    st.subheader("Session")
    if st.button("New conversation"):
        pipeline.memory.reset(st.session_state["session_id"])
        st.session_state["messages"] = []
        st.rerun()

if not pipeline.is_ready():
    st.info("The knowledge base is empty. Upload documents from the sidebar "
            "to get started, then ask a question below.")


def _confidence_badge(conf: float) -> str:
    if conf >= 0.66:
        return f"High confidence ({conf:.2f})"
    if conf >= 0.33:
        return f"Medium confidence ({conf:.2f})"
    return f"Low confidence ({conf:.2f})"


def _save_feedback(question: str, answer: str, helpful: bool) -> None:
    path = settings.feedback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "question": question, "answer": answer, "helpful": helpful,
            "session_id": st.session_state["session_id"],
        }) + "\n")


# ---------------------------------------------------------------- history
for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            st.caption(_confidence_badge(meta["confidence"]))
            if meta["sources"]:
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        st.markdown(f"- **{s['document']}**, page {s['page']}")
            if meta.get("retrieved"):
                with st.expander("Retrieved passages"):
                    for r in meta["retrieved"]:
                        st.markdown(
                            f"**{r['source']}** (p.{r['page']}, score {r['score']:.3f})"
                        )
                        st.caption(r["snippet"])


# ---------------------------------------------------------------- chat input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            result = pipeline.answer(question, st.session_state["session_id"])
        ans = result.answer
        st.markdown(ans.answer)
        st.caption(_confidence_badge(ans.confidence))

        sources = [{"document": s.document, "page": s.page} for s in ans.sources]
        retrieved = [
            {"source": c.source, "page": c.page, "score": c.score, "snippet": c.text[:300]}
            for c in result.retrieved
        ]
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- **{s['document']}**, page {s['page']}")
        if retrieved:
            with st.expander("Retrieved passages"):
                for r in retrieved:
                    st.markdown(f"**{r['source']}** (p.{r['page']}, score {r['score']:.3f})")
                    st.caption(r["snippet"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": ans.answer,
                "meta": {"confidence": ans.confidence, "sources": sources, "retrieved": retrieved},
            }
        )

        # Feedback buttons.
        col1, col2, _ = st.columns([1, 1, 6])
        if col1.button("Helpful", key=f"up-{len(st.session_state['messages'])}"):
            _save_feedback(question, ans.answer, True)
            st.toast("Thanks for the feedback!")
        if col2.button("Not helpful", key=f"down-{len(st.session_state['messages'])}"):
            _save_feedback(question, ans.answer, False)
            st.toast("Thanks — we'll use this to improve.")
