# eval.py
from __future__ import annotations
import json, re, time, sys, traceback
from pathlib import Path
from typing import List, Dict, Any

PRINT_PREFIX = "[eval]"

def p(msg: str):
    print(f"{PRINT_PREFIX} {msg}", flush=True)

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (s or "").lower())

def faithfulness(answer: str, used_texts: List[str]) -> float:
    """
    Very light heuristic: share-of-sentences from the answer that have
    at least one 6+ char substring appearing in the retrieved context.
    Returns 0..1 (higher is more grounded).
    """
    ans = re.split(r"[.\n]", answer or "")
    ans = [a.strip() for a in ans if len(a.strip()) > 0]
    ctx = _normalize(" ".join(used_texts))
    good = 0
    for s in ans:
        s_norm = _normalize(s)
        tokens = [t for t in s_norm.split() if len(t) >= 6]
        ok = any(t in ctx for t in tokens)
        if ok: good += 1
    return (good / len(ans)) if ans else 0.0

def main():
    t0 = time.time()
    p("Starting evaluation…")

    # Imports that rely on your project
    try:
        from rag import retrieve, rerank, answer_from_context
        p("Imported rag.py functions successfully.")
    except Exception as e:
        p("ERROR importing from rag.py:")
        traceback.print_exc()
        sys.exit(1)

    # Seed eval set (edit/expand freely)
    EVAL_QA = [
        {"q": "What is the return window for returns?", "must": ["30", "day"]},
        {"q": "How long do refunds take once an item is returned?", "must": ["5", "7", "business"]},
        {"q": "What is the standard delivery timeline within the UK?", "must": ["3", "5", "business"]},
        {"q": "Are warranty claims accepted for accidental damage?", "must": ["not", "covered"]},
        {"q": "Which products were the top sellers in 2024?", "must": ["top", "product"]},
        {"q": "What were the overall sales trends in 2024?", "must": ["growth", "trend"]},
    ]

    # Quick sanity: index + meta exist
    from utils.io import ARTIFACTS_DIR
    faiss_dir = ARTIFACTS_DIR / "faiss"
    chunks_dir = ARTIFACTS_DIR / "chunks"

    if not (faiss_dir / "index.faiss").exists():
        p(f"ERROR: FAISS index not found at {faiss_dir/'index.faiss'}. Run: python index.py build")
        sys.exit(1)
    if not (faiss_dir / "meta.json").exists():
        p(f"ERROR: meta.json not found at {faiss_dir/'meta.json'}. Run: python index.py embed && python index.py build")
        sys.exit(1)
    if not (chunks_dir / "chunks.jsonl").exists():
        p(f"ERROR: chunks.jsonl not found at {chunks_dir/'chunks.jsonl'}. Run: python ingest.py")
        sys.exit(1)
    p("Artifacts found (index.faiss, meta.json, chunks.jsonl).")

    # Retrieval metrics
    def hit_at_k(query: str, terms: List[str], k: int = 5) -> int:
        hits = retrieve(query, top_k=max(10, k))
        hits = rerank(query, hits, rerank_top_k=k)
        joined = _normalize(" ".join((h.text or "") for h in hits))
        return int(all(t in joined for t in [_normalize(x) for x in terms]))

    def mrr_at_k(query: str, terms: List[str], k: int = 10) -> float:
        hits = retrieve(query, top_k=max(20, k))
        hits = rerank(query, hits, rerank_top_k=k)
        for i, h in enumerate(hits, start=1):
            t = _normalize(h.text or "")
            if all(x in t for x in [_normalize(x) for x in terms]):
                return 1.0 / i
        return 0.0

    p("Computing retrieval metrics (Hit@5, MRR@5)…")
    k_rer = 5
    try:
        hits_sum = 0
        mrr_sum = 0.0
        for item in EVAL_QA:
            hits_sum += hit_at_k(item["q"], item["must"], k=k_rer)
            mrr_sum  += mrr_at_k(item["q"], item["must"], k=k_rer)
        Hit5 = hits_sum / len(EVAL_QA)
        MRR5 = mrr_sum / len(EVAL_QA)
        p(f"Retrieval — Hit@5={Hit5:.2f}, MRR@5={MRR5:.3f} on n={len(EVAL_QA)}")
    except Exception as e:
        p("ERROR while computing retrieval metrics:")
        traceback.print_exc()
        sys.exit(1)

    # Generation + faithfulness sample (first 3 Qs)
    p("Generating answers for 3 samples and computing faithfulness…")
    samples = []
    try:
        for item in EVAL_QA[:3]:
            ans, used = answer_from_context(item["q"], top_k=10, rerank_top_k=k_rer)
            faith = faithfulness(ans, [h.text or "" for h in used[:k_rer]])
            samples.append({
                "q": item["q"],
                "answer_preview": (ans or "")[:300],
                "faithfulness": round(faith, 3),
                "sources": [h.source_path for h in used[:k_rer]]
            })
        p("Samples generated.")
    except Exception as e:
        p("ERROR during generation or faithfulness computation:")
        traceback.print_exc()
        sys.exit(1)

    results = {
        "n": len(EVAL_QA),
        "Hit@5": round(Hit5, 3),
        "MRR@5": round(MRR5, 3),
        "samples": samples,
        "elapsed_sec": round(time.time() - t0, 2)
    }

    out = ARTIFACTS_DIR / "eval" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    p(f"Saved results to: {out}")
    p("Done.")

if __name__ == "__main__":
    main()
