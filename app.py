# app.py
from __future__ import annotations

import os, sys, re, time, json, uuid, subprocess
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure DOCS_DIR points at the repo /docs (works locally & on Streamlit Cloud)
REPO_ROOT = Path(__file__).parent.resolve()
os.environ.setdefault("DOCS_DIR", str(REPO_ROOT / "docs"))

# RAG pipeline entrypoint (safe to import now)
from rag import answer_from_context

# Optional: index builders (we call them if FAISS is missing)
from index import embed_chunks, build_faiss_index

load_dotenv()
st.set_page_config(page_title="Retail Knowledge Assistant", layout="wide")

# ---------- One-time builders (first run on Streamlit Cloud) ----------
def ensure_chunks():
    """
    Build artifacts/chunks/chunks.jsonl if missing by calling ingest.py.
    Shows readable errors in the UI if something goes wrong.
    """
    chunks_path = REPO_ROOT / "artifacts" / "chunks" / "chunks.jsonl"
    if chunks_path.exists():
        return

    # sanity: do we have docs?
    docs_dir = Path(os.environ.get("DOCS_DIR", "") or (REPO_ROOT / "docs"))
    md_files = list(docs_dir.glob("*.md"))
    if not md_files:
        st.error(
            f"No Markdown files found in DOCS_DIR={docs_dir}.\n"
            "Add your .md files under the repo's /docs folder (or set DOCS_DIR in Streamlit Secrets)."
        )
        st.stop()

    (REPO_ROOT / "artifacts" / "chunks").mkdir(parents=True, exist_ok=True)
    with st.spinner("📄 First run: preparing document chunks…"):
        # Use the same Python interpreter Streamlit uses
        proc = subprocess.run(
            [sys.executable, "ingest.py"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            st.error("Ingestion failed.\n\nSTDERR:\n" + (proc.stderr or "")[:2000])
            st.caption("STDOUT:\n" + (proc.stdout or "")[:2000])
            st.stop()

def ensure_index():
    """
    Build artifacts/faiss/index.faiss (+meta/embeddings/ids) if missing.
    Uses in-process functions from index.py for speed & fewer env surprises.
    """
    faiss_dir = REPO_ROOT / "artifacts" / "faiss"
    idx  = faiss_dir / "index.faiss"
    meta = faiss_dir / "meta.json"
    emb  = faiss_dir / "embeddings.npy"
    ids  = faiss_dir / "ids.json"

    if idx.exists() and meta.exists() and emb.exists() and ids.exists():
        return

    with st.spinner("🔧 First run: building vector index…"):
        try:
            embed_chunks()        # reads artifacts/chunks/chunks.jsonl
            build_faiss_index()   # writes FAISS + meta
        except Exception as e:
            st.error(f"Index build failed: {e}")
            st.stop()

# Build what’s needed before we accept queries
ensure_chunks()
ensure_index()

# ---------- Helpers (UI polish) ----------
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

st.sidebar.header("Settings")
provider = st.sidebar.selectbox(
    "Generation provider",
    ["Cohere", "OpenAI", "Offline"],
    index=0 if os.getenv("USE_COHERE", "false").lower() == "true"
         else (1 if os.getenv("USE_OPENAI", "false").lower() == "true" else 2)
)
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

if "history" not in st.session_state:
    st.session_state.history = []
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []

query = st.text_input("Ask about returns, refunds, warranty, delivery, products…", key="query_input")

if query:
    t0 = time.time()
    answer, hits = answer_from_context(query, top_k=top_k, rerank_top_k=rerank_k)
    latency = time.time() - t0
    prov = provider_label()
    st.session_state.history.append((query, answer, hits, latency, prov))

    rq = st.session_state.recent_queries
    if not rq or rq[-1] != query:
        rq.append(query)
    st.session_state.recent_queries = rq[-5:]

    try:
        os.makedirs("artifacts/logs", exist_ok=True)
        with open("artifacts/logs/interactions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
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
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
