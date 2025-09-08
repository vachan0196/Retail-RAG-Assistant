# utils/text.py
from __future__ import annotations
import re
from typing import List, Dict, Any

# --- Optional tiktoken import (falls back to estimator on cloud) ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(s: str) -> int:
        # Real tokenization when tiktoken is available
        return len(_enc.encode(s or ""))

except Exception:
    # Cloud-safe fallback: quick heuristic (~4 chars ≈ 1 token)
    def count_tokens(s: str) -> int:
        if not s:
            return 0
        # Prefer word pieces; then fallback to char-length / 4
        # This keeps chunk sizes in a sensible range without tiktoken.
        words = re.findall(r"\S+", s)
        approx = max(len(words), len(s) // 4)
        return approx

# --- rest of your file below stays the same ---

def markdown_to_chunks(
    text: str,
    *,
    max_tokens: int = 350,
    overlap: int = 40,
    **kwargs
) -> List[str]:
    """
    Simple markdown segmenter with token-budgeted chunks.
    Uses count_tokens() which works with/without tiktoken.

    Accepts alias 'overlap_tokens' for backward-compat.
    """
    # Backward-compat alias
    if "overlap_tokens" in kwargs and isinstance(kwargs["overlap_tokens"], (int, float, str)):
        try:
            overlap = int(kwargs["overlap_tokens"])
        except Exception:
            pass

    # normalize newlines
    text = (text or "").replace("\r\n", "\n")
    # split on headers/blank lines as soft boundaries
    parts = re.split(r"\n{2,}", text)

    chunks: List[str] = []
    buf: List[str] = []
    buf_tok = 0
    for p in parts:
        ptok = count_tokens(p)
        if buf and buf_tok + ptok > max_tokens:
            # flush buffer
            block = "\n\n".join(buf).strip()
            if block:
                chunks.append(block)
            # start new buffer, seed overlap
            if overlap and chunks:
                tail = chunks[-1]
                # take ~overlap tokens from end (roughly on sentences)
                tail_sents = re.split(r"(?<=[.!?])\s+", tail)
                take = []
                t = 0
                for s in reversed(tail_sents):
                    t += count_tokens(s)
                    take.append(s)
                    if t >= overlap:
                        break
                buf = [" ".join(reversed(take)), p]
                buf_tok = count_tokens(buf[0]) + ptok
            else:
                buf = [p]
                buf_tok = ptok
        else:
            buf.append(p)
            buf_tok += ptok

    if buf:
        block = "\n\n".join(buf).strip()
        if block:
            chunks.append(block)

    return chunks
