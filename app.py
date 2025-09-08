# app.py
from __future__ import annotations

import os, sys, re, time, json, uuid, subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# RAG pipeline entrypoint
from rag import answer_from_context

# For bootstrap (build chunks/embeddings/index on first run)
from utils.io import ARTIFACTS_DIR
from index import embed_chunks, build_faiss_index

# ---------- Boot ----------
load_dotenv()
st.set_page_config(page_title="Retail Knowledge Assistant", layout="wide")


# ---------- First-run bootstrap ----------
def ensure_chunks() -> None:
    """
    Make sure artifacts/chunks/chunks.jsonl exists.
    If not, run ingest.py (uses DOCS_DIR) to create it.
    """
    chunks_path = ARTIFACTS_DIR / "chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        docs_dir = os.getenv("DOCS_DIR", "docs")
        if not Path(docs_dir).exists():
            st.error(
                f"❌ No docs found at `{docs_dir}`. "
                "Add your Markdown files to the repo's docs/ folder (or set DOCS_DIR)."
            )
            st.stop()
        with st.spinner("📄 First run: preparing document chunks…"):
            # Call the script so it can read DOCS_DIR and produce chunks.jsonl
            subprocess.run([sys.executable, "ingest.py"], check=True)


def ensure_index() -> None:
    """
    Make sure FAISS index and metadata exist.
    If missing, embed all chunks and build the FAISS index.
    """
    faiss_dir = ARTIFACTS_DIR / "faiss"
    idx = faiss_dir / "index.faiss"
    meta = faiss_dir / "meta.json"
    emb = faiss_dir / "embeddings.npy"

    if not idx.exists() or not meta.exists() or not emb.exists():
        with st.spinner("🔧 First run: building vector index…"):
            embed_chunks()
            build_faiss_index()
        st.success("✅ Index built. Ready to answer!")


# Run bootstrap before rendering the rest of the UI
try:
    ensure_chunks()
    ensure_index()
except subprocess.CalledProcessError as e:
    st.error("❌ Failed to prepare chunks/index. See logs for details.")
    st.stop()
except Exception as e:
    st.error(f"❌ Setup error: {e}")
    st.stop()


# ---------- Helpers ----------
def clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[^a-zA-Z0-9\s\-\&\(\)\.,:]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_answer(text: str) -> str:
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


# ---------- UI ----------
st.title("🛍️ Retail Knowledge Assistant")
st.caption("Ask about returns, refunds, warranty, shipping, and products — answers are grounded in your internal docs with citations.")

# Sidebar: settings
st.sidebar.header("Settings")
provider = st.sidebar.selectbox(
    "Generation provider",
    ["Cohere", "OpenAI", "Offline"],
    index=0 if os.getenv("USE_COHERE", "false").lower() == "true"
         else (1 if os.getenv("USE_OPENAI", "false").lower() == "true" else 2)
)
# reflect provider into env for this session (rag.py reads os.getenv each call)
os.environ["USE_COHERE"] = "true" if provider == "Cohere" else "false"
os.environ["USE_OPENAI"] = "true" if provider == "OpenAI" else "false"

top_k = st.sidebar.slider("Retriever top-k", 5, 20, 10)
rerank_k = st.sidebar.slider("Rerank top-k", 1, 10, 5)

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.history = []
    st.rerun()

st.sidebar.caption(
    f"Provider: {'🟢' if os.getenv('USE_OPENAI')=='true' else '⚪'} OpenAI  |  "
    f"{'🟢' if os.getenv('USE_COHERE')=='true' else '⚪'} Cohere  |  "
    f"{'🟢' if (os.getenv('USE_OPENAI')!='true' and os.getenv('USE_COHERE')!='true') else '⚪'} Offline"
)

st.sidebar.subheader("Recent searches")
for i, qprev in enumerate(reversed(st.session_state.get("recent_queries", [])), 1):
    label = f"{i}. {qprev[:60]}{'…' if len(qprev) > 60 else ''}"
    if st.sidebar.button(label, key=f"recent_{i}"):
        st.session_state["query_input"] = qprev
        st.rerun()
if not st.session_state.get("recent_queries"):
    st.sidebar.caption("No recent searches yet.")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

# Query box
query = st.text_input(
    "Ask about returns, refunds, warranty, delivery, products…",
    key="query_input"
)

# Answer
if query:
    try:
        t0 = time.time()
        answer, hits = answer_from_context(query, top_k=top_k, rerank_top_k=rerank_k)
        latency = time.time() - t0
        prov = provider_label()
        st.session_state.history.append((query, answer, hits, latency, prov))

        # recent searches (keep last 5)
        rq = st.session_state.recent_queries
        if not rq or rq[-1] != query:
            rq.append(query)
        st.session_state.recent_queries = rq[-5:]

        # lightweight logging
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
            pass
    except FileNotFoundError:
        st.warning("Index is still building. Please wait a moment and try again.")
        st.stop()
    except Exception as e:
        st.error(f"Generation error: {e}")
        st.stop()

# Render chat (newest first)
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
