# utils/io.py
from __future__ import annotations
import os, json, csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHUNKS_DIR = ARTIFACTS_DIR / "chunks"
RAW_DIR = ARTIFACTS_DIR / "raw"

DOCS_DIR = os.getenv("DOCS_DIR", r"D:\Ai projects\New folder (2)\docs")

def ensure_dirs() -> None:
    for p in [ARTIFACTS_DIR, CHUNKS_DIR, RAW_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def path(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

def docs_path(*parts) -> Path:
    return Path(DOCS_DIR).joinpath(*parts)

def read_text(p: Path, encoding: str = "utf-8") -> str:
    return p.read_text(encoding=encoding, errors="ignore")

def write_jsonl(items: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def read_document_index(csv_path: Path) -> List[Dict[str, Any]]:
    """
    document_index.csv columns (flexible, typical):
      filename, title, doc_type, product_ids, tags
    """
    rows: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize some fields
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if "product_ids" in r and r["product_ids"]:
                r["product_ids"] = [x.strip() for x in r["product_ids"].split("|") if x.strip()]
            else:
                r["product_ids"] = []
            if "tags" in r and r["tags"]:
                r["tags"] = [x.strip() for x in r["tags"].split("|") if x.strip()]
            else:
                r["tags"] = []
            rows.append(r)
    return rows

def list_markdown_files(dir_path: Path) -> List[Path]:
    exts = {".md", ".markdown", ".mdown"}
    return sorted([p for p in dir_path.glob("*.md") if p.suffix.lower() in exts])
