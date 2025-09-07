# rag.py
from __future__ import annotations

import os, json, argparse, math, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from utils.io import ARTIFACTS_DIR
from utils.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context

# ---------- boot / secrets helpers ----------
load_dotenv()

# Prefer Streamlit secrets if running under Streamlit Cloud; fallback to env
try:
    import streamlit as st
    _SECRETS = st.secrets  # Mapping-like
except Exception:
    _SECRETS = {}

def env(key: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then os.getenv."""
    val = _SECRETS.get(key, None)
    if val is None:
        return os.getenv(key, default)
    # Streamlit secrets values can be non-str; normalize
    return str(val)

def env_bool(key: str, default: bool = False) -> bool:
    val = env(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "y", "on")

# ---------- model + cache config ----------
EMBED_MODEL   = env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL= env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Local HF cache (helps on Streamlit Cloud to avoid repeated downloads)
CACHE_DIR = env("HF_CACHE_DIR", "artifacts/models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
# Respect if huggingface_hub reads HF_HOME / TRANSFORMERS_CACHE
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", CACHE_DIR)

FAISS_DIR  = ARTIFACTS_DIR / "faiss"
INDEX_PATH = FAISS_DIR / "index.faiss"
EMB_PATH   = FAISS_DIR / "embeddings.npy"   # not strictly needed here
META_PATH  = FAISS_DIR / "meta.json"
IDS_PATH   = FAISS_DIR / "ids.json"

@dataclass
class Hit:
    id: str
    score: float
    title: str
    doc_type: str
    header_path: str
    source_path: str
    text: Optional[str] = None

def _load_index_and_meta():
    import faiss
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Build it via `python index.py build`.")
    if not META_PATH.exists() or not IDS_PATH.exists():
        raise FileNotFoundError("Missing meta/ids. Run embedding + build steps.")

    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with IDS_PATH.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    id_to_meta = {m["id"]: m for m in meta["meta"]}
    return index, id_to_meta, ids

def _load_chunks_text_map() -> Dict[str, str]:
    chunks_path = ARTIFACTS_DIR / "chunks" / "chunks.jsonl"
    mp: Dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mp[row["id"]] = row["text"]
    return mp

def retrieve(
    query: str,
    top_k: int = 10,
    filter_doc_types: Optional[List[str]] = None,
    filter_product_ids: Optional[List[str]] = None
) -> List[Hit]:
    """First-stage semantic retrieval (FAISS)."""
    import faiss
    index, id_to_meta, ids = _load_index_and_meta()

    emb_model = SentenceTransformer(EMBED_MODEL, cache_folder=CACHE_DIR)
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # Over-fetch to allow optional filtering
    D, I = index.search(qv, top_k * 3)
    prelim: List[Hit] = []
    id_text = _load_chunks_text_map()

    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunk_id = ids[idx]
        m = id_to_meta.get(chunk_id)
        if not m:
            continue
        if filter_doc_types and m.get("doc_type") not in filter_doc_types:
            continue

        prelim.append(Hit(
            id=chunk_id,
            score=float(dist),
            title=m.get("title", ""),
            doc_type=m.get("doc_type", ""),
            header_path=m.get("header_path", ""),
            source_path=m.get("source_path", ""),
            text=id_text.get(chunk_id, "")
        ))
        if len(prelim) >= top_k * 2:
            break

    prelim.sort(key=lambda h: h.score, reverse=True)
    return prelim[:top_k]

def rerank(query: str, hits: List[Hit], rerank_top_k: int = 5) -> List[Hit]:
    """CrossEncoder reranking: re-score (query, chunk_text) pairs."""
    if not hits:
        return []
    reranker = CrossEncoder(RERANKER_MODEL, cache_folder=CACHE_DIR)
    pairs = [(query, h.text or "") for h in hits]
    scores = reranker.predict(pairs)  # logits
    for h, s in zip(hits, scores):
        prob = 1 / (1 + math.exp(-float(s)))  # for display only
        h.score = float(prob)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:rerank_top_k]

def retrieve_and_rerank(query: str, top_k: int = 10, rerank_top_k: int = 5, filter_doc_types: Optional[List[str]] = None) -> List[Hit]:
    base = retrieve(query, top_k=top_k, filter_doc_types=filter_doc_types)
    return rerank(query, base, rerank_top_k=rerank_top_k)

def _top_chunks_for_prompt(hits, max_chars: int = 5000):
    out = []
    total = 0
    for h in hits:
        txt = h.text or ""
        snippet = txt if len(txt) <= 1200 else txt[:1200]
        item = {"title": h.title, "header_path": h.header_path, "text": snippet}
        block = f"{h.title} §{h.header_path}\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        out.append(item)
        total += len(block)
    return out

# ---------- LLM providers ----------
def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=env("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def _call_cohere(system_prompt: str, user_prompt: str) -> str:
    import cohere
    api_key = env("COHERE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing COHERE_API_KEY")
    co = cohere.Client(api_key)
    # Try classic signature
    try:
        resp = co.chat(model="command-r", preamble=system_prompt, message=user_prompt, temperature=0.2)
        if getattr(resp, "text", None):
            return resp.text.strip()
    except TypeError:
        pass
    except Exception:
        raise
    # Try newer signature
    try:
        resp = co.chat(
            model="command-r",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.2,
        )
        if getattr(resp, "text", None):
            return resp.text.strip()
    except TypeError:
        # Final fallback: generate()
        text = f"{system_prompt}\n\n{user_prompt}"
        gen = co.generate(model="command", prompt=text, temperature=0.2)
        return gen.generations[0].text.strip()
    return ""

def answer_from_context(query: str, top_k: int = 10, rerank_top_k: int = 5) -> Tuple[str, list]:
    hits0 = retrieve(query, top_k=top_k)
    hits = rerank(query, hits0, rerank_top_k=rerank_top_k)
    ctx_items = _top_chunks_for_prompt(hits)
    ctx_str = format_context(ctx_items)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=query, context=ctx_str)

    use_openai = env_bool("USE_OPENAI", False) and bool(env("OPENAI_API_KEY", ""))
    use_cohere = env_bool("USE_COHERE", True)  and bool(env("COHERE_API_KEY", ""))

    if use_openai:
        try:
            ans = _call_openai(SYSTEM_PROMPT, user_prompt)
            print("[gen] OpenAI path used.")
            return ans, hits
        except Exception as e:
            print(f"[gen] OpenAI failed -> will try Cohere/offline. Reason: {repr(e)}")

    if use_cohere:
        try:
            ans = _call_cohere(SYSTEM_PROMPT, user_prompt)
            print("[gen] Cohere path used.")
            return ans, hits
        except Exception as e:
            print(f"[gen] Cohere failed -> falling back. Reason: {repr(e)}")

    # ---------- Offline fallback: extract time windows + crisp answer ----------
    time_patterns = [
        r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\s*(business\s+days|days|working\s+days)\b",
        r"\b(\d{1,3})\s*(business\s+days|days|working\s+days|calendar\s+days)\b",
        r"\b(within|up to)\s+(\d{1,3})\s*(business\s+days|days|working\s+days)\b",
    ]
    def find_time_windows(text: str):
        found = []
        for pat in time_patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                found.append(m.group(0))
        return found

    findings = []
    for h in hits:
        tw = find_time_windows(h.text or "")
        if tw:
            hdr = f" §{h.header_path}" if h.header_path else ""
            findings.append({"times": tw, "title": h.title, "hdr": hdr, "src": h.source_path})
        if len(findings) >= 4:
            break

    def rank_finding(f):
        title = (f["title"] or "").lower()
        hdr   = (f["hdr"] or "").lower()
        score = 0
        if "return" in hdr or "return" in title: score += 3
        if "process" in hdr: score += 2
        if "policy" in title: score += 1
        if "summary" in hdr: score -= 1
        return score

    findings.sort(key=rank_finding, reverse=True)

    if findings:
        primary = findings[0]["times"][0]
        cite = f"[{findings[0]['title']}{findings[0]['hdr']}]"
        answer = f"**Return window:** {primary}. {cite}"
        if len(findings) > 1:
            extras = []
            for f in findings[1:]:
                extras.append(f"- {f['times'][0]}  [{f['title']}{f['hdr']}]")
            if extras:
                answer += "\n\n**Related timings:**\n" + "\n".join(extras)
        return answer, hits

    # Fallback to quick bullets
    lines = []
    for h in hits:
        hdr = f" §{h.header_path}" if h.header_path else ""
        cite = f"[{h.title}{hdr}]"
        snippet = (h.text or "").split(". ")
        if snippet:
            lines.append(f"- {snippet[0].strip()}. {cite}")
        if len(lines) >= 4:
            break
    if not lines:
        return "I don’t know based on the provided documents.", hits
    answer = "Here’s what I found:\n" + "\n".join(lines)
    return answer, hits

# ---------- CLI ----------
def _print_hits(query: str, hits: List[Hit], header: str):
    print(f"\nQuery: {query}")
    print(header)
    for i, h in enumerate(hits, start=1):
        hdr = f" §{h.header_path}" if h.header_path else ""
        print(f"#{i}  {h.title}{hdr} [{h.doc_type}]  | score={h.score:.4f}")
        print(f"    id={h.id}  |  src={h.source_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, help="query")
    ap.add_argument("--k", type=int, default=10, help="retriever top-k")
    ap.add_argument("--rk", type=int, default=5, help="rerank top-k")
    ap.add_argument("--only", choices=["retriever", "reranked"], help="debug retrieval modes")
    ap.add_argument("--cmd", choices=["search", "answer"], default="answer")
    args = ap.parse_args()

    if args.cmd == "search":
        if not args.q:
            raise SystemExit("Provide a query with --q")
        if args.only == "retriever":
            hits = retrieve(args.q, top_k=args.k)
            _print_hits(args.q, hits, header="Top hits (semantic retriever only)")
        else:
            hits0 = retrieve(args.q, top_k=args.k)
            _print_hits(args.q, hits0, header="Retriever results (pre-rerank)")
            hits = rerank(args.q, hits0, rerank_top_k=args.rk)
            _print_hits(args.q, hits, header="Top hits (after CrossEncoder rerank)")
    else:
        if not args.q:
            raise SystemExit("Provide a query with --q")
        ans, used = answer_from_context(args.q, top_k=args.k, rerank_top_k=args.rk)
        print("\n=== Answer ===")
        print(ans)
        print("\n=== Sources ===")
        for h in used[:args.rk]:
            hdr = f" §{h.header_path}" if h.header_path else ""
            print(f"- {h.title}{hdr}  ({h.source_path})")

if __name__ == "__main__":
    main()
