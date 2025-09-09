# app.py
from __future__ import annotations

import os, re, time, json, uuid
import streamlit as st
from dotenv import load_dotenv

# Import the RAG pipeline entrypoint (safe: faiss is imported lazily inside functions)
from rag import answer_from_context

# ---------- Boot ----------
load_dotenv()
st.set_page_config(page_title="Retail Knowledge Assistant", layout="wide")

# ---------- FAISS availability check ----------
try:
    import faiss  # noqa: F401
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

# ---------- Helpers ----------
def clean(s: str) -> str:
    """Sanitize titles/headers for clean display."""
    if not s:
        return ""
    s = re.sub(r"[^a-zA-Z0-9\s\-\&\(\)\.,:]", "", s)  # strip odd chars like $, #, *, etc.
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_answer(text: str) -> str:
    """
    - Normalize inline citations: [Title §Header] -> [Title > Header]
    - Remove stray '$'
    - Tidy whitespace/newlines
    """
    if not text:
        return ""
    def _cit_sub(m: re.Match) -> str:
        left = m.group(1).strip()
        right = m.group(2).strip()
        return f"[{left} > {right}]"
    text = re.sub(r"\[([^\[\]]+?)\s*§\s*([^\[\]]+?)\]", _cit_sub, text)
    text = text.replace("$", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def provider_label() -> str:
    if os.getenv("USE_OPENAI") == "true":
        return "openai"
    if os.getenv("USE_COHERE") == "true":
        return "cohere"
    return "offline"

# ---------- UI Header ----------
st.title("🛍️ Retail Knowledge Assistant")
st.caption("Ask about returns, refunds, warranty, shipping, and products — answers are grounded in your internal docs with citations.")

# If FAISS is missing, show a helpful error and stop early
if not FAISS_OK:
    st.error(
        "⚠️ FAISS library is not installed or incompatible.\n\n"
        "Please ensure your environment installs a supported version, e.g. "
        "`faiss-cpu==1.8.0` or `faiss-cpu==1.9.0` (depending on your Python version), "
        "then restart the app."
    )
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")

# Provider selection (overrides env toggles at runtime)
provider = st.sidebar.selectbox(
    "Generation provider",
    ["Cohere", "OpenAI", "Offline"],
    index=0 if os.getenv("USE_COHERE", "false").lower() == "true"
         else (1 if os.getenv("USE_OPENAI", "false").lower() == "true" else 2)
)
# Reflect provider into env for this session (rag.py reads os.getenv each call)
os.environ["USE_COHERE"] = "true" if provider == "Cohere" else "false"
os.environ["USE_OPENAI"] = "true" if provider == "OpenAI" else "false"

top_k = st.sidebar.slider("Retriever top-k", 5, 20, 10)
rerank_k = st.sidebar.slider("Rerank top-k", 1, 10, 5)

st.sidebar.divider()
# Clear chat should NOT clear recent searches
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.history = []
    st.rerun()

st.sidebar.caption(
    f"Provider: {'🟢' if os.getenv('USE_OPENAI')=='true' else '⚪'} OpenAI  |  "
    f"{'🟢' if os.getenv('USE_COHERE')=='true' else '⚪'} Cohere  |  "
    f"{'🟢' if (os.getenv('USE_OPENAI')!='true' and os.getenv('USE_COHERE')!='true') else '⚪'} Offline"
)

st.sidebar.subheader("Recent searches")
# Show newest first, click to re-run
for i, qprev in enumerate(reversed(st.session_state.get("recent_queries", [])), 1):
    label = f"{i}. {qprev[:60]}{'…' if len(qprev) > 60 else ''}"
    if st.sidebar.button(label, key=f"recent_{i}"):
        st.session_state["query_input"] = qprev
        st.rerun()
if not st.session_state.get("recent_queries"):
    st.sidebar.caption("No recent searches yet.")

# ---------- Session state ----------
if "history" not in st.session_state:
    # Entries: (query, answer, hits, latency_sec, provider_str)
    st.session_state.history = []
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []  # persisted even when chat is cleared

# ---------- Query input ----------
query = st.text_input(
    "Ask about returns, refunds, warranty, delivery, products…",
    key="query_input"
)

if query:
    t0 = time.time()
    answer, hits = answer_from_context(query, top_k=top_k, rerank_top_k=rerank_k)
    latency = time.time() - t0
    prov = provider_label()
    st.session_state.history.append((query, answer, hits, latency, prov))

    # Update recent searches (keep max 5; avoid duplicating the same last query)
    rq = st.session_state.recent_queries
    if not rq or rq[-1] != query:
        rq.append(query)
    st.session_state.recent_queries = rq[-5:]

    # Lightweight logging (LLMOps-lite)
    try:
        os.makedirs("artifacts/logs", exist_ok=True)
        log_path = "artifacts/logs/interactions.jsonl"
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "query": query,
            "answer_preview": (answer or "")[:300],
            "provider": prov,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "latency_sec": round(latency, 3),
            "sources": [{
                "id": h.id,
                "title": h.title,
                "header": h.header_path,
                "path": h.source_path
            } for h in (hits or [])[:rerank_k]]
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Keep UI resilient even if logging fails
        pass

# ---------- Render chat (newest first) ----------
for q, a, hits, t, prov in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(clean_answer(a))

    with st.expander("Sources"):
        for h in (hits or [])[:rerank_k]:
            hdr = f" > {clean(h.header_path)}" if h.header_path else ""
            st.write(f"- {clean(h.title)}{hdr}")
            st.code(h.source_path, language="text")

    st.caption(f"Answered in {t:.2f}s • top-k={top_k} • rerank={rerank_k} • provider={prov}")
    st.divider()
