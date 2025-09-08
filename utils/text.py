# utils/text.py
from __future__ import annotations
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import tiktoken

# Use a stable tokenizer for budgeting (OpenAI cl100k_base works well)
_TOK = tiktoken.get_encoding("cl100k_base")

def count_tokens(s: str) -> int:
    if not s:
        return 0
    return len(_TOK.encode(s))

def stable_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _nearest_header_path(lines: List[str], idx: int) -> str:
    """
    Walk backwards from a line index to find the nearest preceding markdown header(s),
    and build a simple 'H1 > H2 > H3' style path. This is lightweight but good enough
    for citations.
    """
    h1 = h2 = h3 = ""
    for j in range(idx, -1, -1):
        line = lines[j].strip()
        if line.startswith("### "):
            if not h3:
                h3 = line[4:].strip()
        elif line.startswith("## "):
            if not h2:
                h2 = line[3:].strip()
        elif line.startswith("# "):
            if not h1:
                h1 = line[2:].strip()
        if h1 and h2 and h3:
            break
    parts = [p for p in (h1, h2, h3) if p]
    return " > ".join(parts)

def markdown_to_chunks(
    text: str,
    *,
    max_tokens: int = 900,
    overlap_tokens: int = 150
) -> List[Dict[str, Any]]:
    """
    Chunk markdown into token-budgeted pieces, preserving an approximate header_path
    for each chunk (used by ingest.py and citations later).
    Returns: List[{"header_path": str, "text": str}]
    """
    if not text:
        return []

    # Normalize newlines and split to lines for header scanning
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")

    # Soft segment candidates: blank-line separated blocks
    blocks: List[Tuple[int, str]] = []  # (start_line_index, block_text)
    buf = []
    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            if buf:
                blocks.append((start_idx, "\n".join(buf).strip()))
                buf = []
            start_idx = i + 1
        else:
            if not buf:
                start_idx = i
            buf.append(ln)
    if buf:
        blocks.append((start_idx, "\n".join(buf).strip()))

    chunks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_tok = 0
    cur_start = 0  # line index of first block in current buffer

    def _flush_with_header(end_line_idx: int):
        nonlocal cur, cur_tok, cur_start
        if not cur:
            return
        block = "\n\n".join(cur).strip()
        if block:
            header_path = _nearest_header_path(lines, end_line_idx)
            chunks.append({
                "header_path": header_path,
                "text": block
            })
        cur = []
        cur_tok = 0

    for (blk_start, blk_txt) in blocks:
        btok = count_tokens(blk_txt)
        if cur and cur_tok + btok > max_tokens:
            # flush current chunk
            _flush_with_header(blk_start)
            # seed overlap from the end of the last chunk
            if overlap_tokens and chunks:
                tail = chunks[-1]["text"]
                # take ~overlap tokens from tail end (split on sentence-ish boundaries)
                sents = re.split(r"(?<=[.!?])\s+", tail)
                take = []
                t = 0
                for s in reversed(sents):
                    t += count_tokens(s)
                    take.append(s)
                    if t >= overlap_tokens:
                        break
                seed = " ".join(reversed(take)).strip()
                if seed:
                    cur = [seed, blk_txt]
                    cur_tok = count_tokens(seed) + btok
                    cur_start = blk_start
                else:
                    cur = [blk_txt]
                    cur_tok = btok
                    cur_start = blk_start
            else:
                cur = [blk_txt]
                cur_tok = btok
                cur_start = blk_start
        else:
            if not cur:
                cur_start = blk_start
            cur.append(blk_txt)
            cur_tok += btok

    if cur:
        _flush_with_header(len(lines) - 1)

    return chunks
