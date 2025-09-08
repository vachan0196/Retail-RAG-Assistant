# ingest.py
from __future__ import annotations
import os, argparse, json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from utils.io import PROJECT_ROOT, CHUNKS_DIR
from utils.text import markdown_to_chunks, count_tokens, stable_id

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(PROJECT_ROOT / "docs")))

def build_chunks_for_file(p: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = p.read_text(encoding="utf-8")
    # original call style: uses overlap_tokens
    chunks = markdown_to_chunks(text, max_tokens=900, overlap_tokens=150)

    rows = []
    for i, ch in enumerate(chunks):
        # ch is a dict with header_path & text (from utils.text original)
        chunk_id = stable_id(f"{p.name}|{i}|{ch['header_path'][:120]}")
        rows.append({
            "id": chunk_id,
            "title": meta.get("title", p.stem),
            "doc_type": meta.get("doc_type", "unknown"),
            "header_path": ch.get("header_path", ""),
            "source_path": str(p),
            "text": ch.get("text", ""),
            "n_tokens": count_tokens(ch.get("text", "")),
        })
    return rows

def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"Docs dir not found: {DOCS_DIR}")

    files = sorted(list(DOCS_DIR.glob("*.md")))
    if not files:
        raise SystemExit("No Markdown files found in your DOCS_DIR. Check .env -> DOCS_DIR.")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / "chunks.jsonl"

    # optional metadata map (document_index.csv), but we’ll keep it simple here
    def infer_meta(path: Path) -> Dict[str, Any]:
        name = path.stem.lower()
        if "faq" in name:
            dt = "faq"
            title = "Faq"
        elif "policy_returns" in name or "returns_refunds" in name:
            dt = "policy"
            title = "Policy Returns Refunds"
        elif "shipping" in name or "delivery" in name:
            dt = "policy"
            title = "Policy Shipping Delivery"
        elif "warranty" in name:
            dt = "policy"
            title = "Policy Warranty"
        elif "catalogue" in name or "catalog" in name:
            dt = "product"
            title = "Product Catalogue"
        elif "annual_sales" in name:
            dt = "summary"
            title = "Annual Sales Summary 2024"
        else:
            dt = "unknown"
            title = path.stem
        return {"doc_type": dt, "title": title}

    total = 0
    per_file = {}
    with out_path.open("w", encoding="utf-8") as f:
        for p in files:
            meta = infer_meta(p)
            rows = build_chunks_for_file(p, meta)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
            per_file[p.name] = len(rows)

    print(f"✅ Ingestion complete: {total} chunks -> {out_path}")
    for k, v in per_file.items():
        print(f"  - {k}: {v} chunks")

if __name__ == "__main__":
    main()
