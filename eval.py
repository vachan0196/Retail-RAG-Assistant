# eval.py
from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import List, Dict, Any
from rag import retrieve, rerank, answer_from_context

ART_DIR = Path("artifacts")
(ART_DIR / "eval").mkdir(parents=True, exist_ok=True)

# --- Seed eval set (edit/extend freely) ---
EVAL_QA = [
    {"q": "What is the return window for returns?", "must": ["30", "day"]},
    {"q": "How long do refunds take once an item is returned?", "must": ["5", "7", "business"]},
    {"q": "What is the standard delivery timeline within the UK?", "must": ["3", "5", "business"]},
    {"q": "Are warranty claims accepted for accidental damage?", "must": ["not", "covered"]},
    {"q": "Which products were the top sellers in 2024?", "must": ["top", "product"]},
    {"q": "What were the overall sales trends in 2024?", "must": ["growth", "trend"]},
]

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (s or "").lower())

def hit_at_k(query: str, terms: List[str], k: int = 5) -> int:
    """Return 1 if all 'terms' appear in the concatenated top-k reranked texts."""
    hits = retrieve(query, top_k=max(10, k))
    hits = rerank(query, hits, rerank_top_k=k)
    joined = _normalize(" ".join((h.text or "") for h in hits))
    return int(all(t in joined for t in [_normalize(x) for x in terms]))

def mrr_at_k(query: str, terms: List[str], k: int = 10) -> float:
    """Return 1/rank of first hit containing all 'terms'; 0 if none."""
    hits = retrieve(query, top_k=max(20, k))
    hits = rerank(query, hits, rerank_top_k=k)
    for i, h in enumerate(hits, start=1):
        t = _normalize(h.text or "")
        if all(x in t for x in [_normalize(x) for x in terms]):
            return 1.0 / i
    return 0.0

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
        # take mid-length substrings
        tokens = [t for t in s_norm.split() if len(t) >= 6]
        ok = any(t in ctx for t in tokens)
        if ok: good += 1
    return (good / len(ans)) if ans else 0.0

def main():
    k_retr = 5
    k_rer  = 5
    t0 = time.time()

    # Retrieval metrics
    hits = sum(hit_at_k(x["q"], x["must"], k=k_rer) for x in EVAL_QA)
    mrr  = sum(mrr_at_k(x["q"], x["must"], k=k_rer) for x in EVAL_QA) / len(EVAL_QA)

    # Generation + faithfulness sample (first 3 Qs)
    gens = []
    for item in EVAL_QA[:3]:
        ans, used = answer_from_context(item["q"], top_k=10, rerank_top_k=k_rer)
        faith = faithfulness(ans, [h.text or "" for h in used[:k_rer]])
        gens.append({
            "q": item["q"],
            "answer_preview": (ans or "")[:300],
            "faithfulness": round(faith, 3),
            "sources": [h.source_path for h in used[:k_rer]]
        })

    results = {
        "n": len(EVAL_QA),
        "Hit@5": hits / len(EVAL_QA),
        "MRR@5": round(mrr, 3),
        "samples": gens,
        "elapsed_sec": round(time.time() - t0, 2)
    }

    out = ART_DIR / "eval" / "results.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Retrieval metrics:", {k: results[k] for k in ["Hit@5","MRR@5","n"]})
    print(f"Faithfulness samples saved to: {out}")

if __name__ == "__main__":
    main()
