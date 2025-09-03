# index.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.io import PROJECT_ROOT, CHUNKS_DIR, ARTIFACTS_DIR

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- New: local cache for HF models (prevents 429 + speeds up) ----
CACHE_DIR = os.getenv("HF_CACHE_DIR", "artifacts/models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

FAISS_DIR = ARTIFACTS_DIR / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
EMB_PATH = FAISS_DIR / "embeddings.npy"
META_PATH = FAISS_DIR / "meta.json"
INDEX_PATH = FAISS_DIR / "index.faiss"
IDS_PATH = FAISS_DIR / "ids.json"

# ---------- Utilities ----------
def read_chunks(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# ---------- Step 2 (already done): Embeddings ----------
def embed_chunks() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks not found: {CHUNKS_PATH}. Run ingest.py first.")

    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = read_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    meta = [{
        "id": c["id"],
        "title": c["title"],
        "doc_type": c["doc_type"],
        "header_path": c["header_path"],
        "source_path": c["source_path"]
    } for c in chunks]

    print(f"Loading embedding model: {EMBED_MODEL}")
    # ---- Changed: pass cache_folder so model is stored locally ----
    model = SentenceTransformer(EMBED_MODEL, cache_folder=CACHE_DIR)

    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(EMB_PATH, embeddings)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "model": EMBED_MODEL,
            "num_vectors": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "ids": ids,
            "meta": meta
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved embeddings → {EMB_PATH}")
    print(f"✅ Saved metadata   → {META_PATH}")
    print(f"Shape: {embeddings.shape}")

# ---------- Step 3: Build FAISS index ----------
def build_faiss_index() -> None:
    import faiss  # import here to fail fast if missing

    if not EMB_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Embeddings/meta not found. Run `python index.py embed` first.")

    X = np.load(EMB_PATH)
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    n, d = X.shape
    print(f"Building FAISS index: {n} vectors, dim={d}")

    # Inner Product index (works as cosine with normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(X.astype(np.float32))

    faiss.write_index(index, str(INDEX_PATH))
    with IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta["ids"], f)

    print(f"✅ Saved FAISS index → {INDEX_PATH}")
    print(f"✅ Saved IDs map     → {IDS_PATH}")

# ---------- Quick search (smoke test) ----------
def search(query: str, top_k: int = 5) -> None:
    import faiss

    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Run `python index.py build` first.")

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    ids = json.load(open(IDS_PATH, "r", encoding="utf-8"))

    index = faiss.read_index(str(INDEX_PATH))

    # Encode query
    # ---- Changed: cache_folder here too ----
    model = SentenceTransformer(EMBED_MODEL, cache_folder=CACHE_DIR)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    D, I = index.search(qv, top_k)
    hits = []
    id_to_meta = {m["id"]: m for m in meta["meta"]}

    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx == -1:
            continue
        chunk_id = ids[idx]
        m = id_to_meta.get(chunk_id, {})
        hits.append({
            "rank": rank,
            "score": float(dist),
            "id": chunk_id,
            "title": m.get("title"),
            "doc_type": m.get("doc_type"),
            "header_path": m.get("header_path"),
            "source_path": m.get("source_path")
        })

    print(f"\nQuery: {query}")
    for h in hits:
        hdr = f" §{h['header_path']}" if h.get("header_path") else ""
        print(f"#{h['rank']}  {h['title']}{hdr}  | score={h['score']:.4f}")
        print(f"    id={h['id']}  |  src={h['source_path']}")
    if not hits:
        print("No hits.")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "build", "search"], help="embed chunks, build faiss index, or search")
    ap.add_argument("--q", type=str, help="query text (for search)")
    ap.add_argument("--k", type=int, default=5, help="top-k results (for search)")
    args = ap.parse_args()

    if args.cmd == "embed":
        embed_chunks()
    elif args.cmd == "build":
        build_faiss_index()
    elif args.cmd == "search":
        if not args.q:
            raise SystemExit("Provide a query with --q \"your question\"")
        search(args.q, args.k)

if __name__ == "__main__":
    main()
