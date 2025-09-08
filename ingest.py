# ingest.py
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from utils.io import (
    ensure_dirs, docs_path, list_markdown_files, read_text,
    read_document_index, CHUNKS_DIR, PROJECT_ROOT
)
from utils.text import markdown_to_chunks, count_tokens

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_metadata_index() -> Dict[str, Dict[str, Any]]:
    """
    Build a map from filename -> metadata from document_index.csv (if present).
    """
    csv_path = docs_path("document_index.csv")
    rows = read_document_index(csv_path)
    meta_by_file: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        filename = r.get("filename") or ""
        if filename:
            meta_by_file[filename] = r
    return meta_by_file

def build_chunks_for_file(p: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = read_text(p)
    chunks = markdown_to_chunks(text, max_tokens=900, overlap=150)
    out: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        chunk_id = stable_id(f"{p.name}|{i}|{ch['header_path'][:120]}")
        record = {
            "id": chunk_id,
            "doc_filename": p.name,
            "source_path": str(p),
            "title": meta.get("title") or p.stem.replace("_", " ").title(),
            "doc_type": meta.get("doc_type") or infer_doc_type(p.name),
            "product_ids": meta.get("product_ids") or [],
            "tags": meta.get("tags") or [],
            "header_path": ch["header_path"],
            "text": ch["chunk_text"],
            "n_tokens": count_tokens(ch["chunk_text"]),
        }
        out.append(record)
    return out

def infer_doc_type(filename: str) -> str:
    fn = filename.lower()
    if "policy" in fn: return "policy"
    if "faq" in fn: return "faq"
    if "catalogue" in fn or "catalog" in fn: return "catalogue"
    if "annual" in fn or "summary" in fn: return "report"
    return "other"

def main():
    ensure_dirs()
    meta_index = load_metadata_index()
    md_files = list_markdown_files(docs_path())

    if not md_files:
        print("No Markdown files found in your DOCS_DIR. Check .env -> DOCS_DIR.")
        return

    all_chunks: List[Dict[str, Any]] = []
    total_tokens = 0
    file_stats = []

    for p in md_files:
        base = p.name
        meta = meta_index.get(base, {})
        chunks = build_chunks_for_file(p, meta)
        all_chunks.extend(chunks)
        total_tokens += sum(c["n_tokens"] for c in chunks)
        file_stats.append((base, len(chunks)))

    # Write a single consolidated JSONL
    out_path = CHUNKS_DIR / "chunks.jsonl"
    from utils.io import write_jsonl
    write_jsonl(all_chunks, out_path)

    print(f"✅ Ingestion complete: {len(all_chunks)} chunks -> {out_path}")
    for base, n in file_stats:
        print(f"  - {base}: {n} chunks")
    print(f"≈ Total tokens across chunks: {total_tokens:,}")

if __name__ == "__main__":
    main()
