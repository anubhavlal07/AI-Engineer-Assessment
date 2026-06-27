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
from src.rag_pipeline import get_pipeline

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")
settings = get_settings()


# ---------------------------------------------------------------- auth gate
def _check_password() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("📚 Enterprise Knowledge Assistant")
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

st.title("📚 Enterprise Knowledge Assistant")

with st.sidebar:
    st.subheader("Session")
    st.write(f"Indexed chunks: **{pipeline._vector_store.count()}**")
    st.write(f"Index ready: **{pipeline.is_ready()}**")
    if st.button("🔄 New conversation"):
        pipeline.memory.reset(st.session_state["session_id"])
        st.session_state["messages"] = []
        st.rerun()

if not pipeline.is_ready():
    st.warning("Knowledge base is empty. Add documents to `data/raw/` and run "
               "`python -m scripts.ingest`.")


def _confidence_badge(conf: float) -> str:
    if conf >= 0.66:
        return f"🟢 High confidence ({conf:.2f})"
    if conf >= 0.33:
        return f"🟡 Medium confidence ({conf:.2f})"
    return f"🔴 Low confidence ({conf:.2f})"


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
                with st.expander("📎 Sources"):
                    for s in meta["sources"]:
                        st.markdown(f"- **{s['document']}**, page {s['page']}")
            if meta.get("retrieved"):
                with st.expander("🔍 Retrieved passages"):
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
            with st.expander("📎 Sources"):
                for s in sources:
                    st.markdown(f"- **{s['document']}**, page {s['page']}")
        if retrieved:
            with st.expander("🔍 Retrieved passages"):
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
        if col1.button("👍", key=f"up-{len(st.session_state['messages'])}"):
            _save_feedback(question, ans.answer, True)
            st.toast("Thanks for the feedback!")
        if col2.button("👎", key=f"down-{len(st.session_state['messages'])}"):
            _save_feedback(question, ans.answer, False)
            st.toast("Thanks — we'll use this to improve.")
