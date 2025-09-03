# utils/text.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
import tiktoken

# Use OpenAI tokenizer for stable token estimates (even if you don't call OpenAI)
ENC = tiktoken.get_encoding("cl100k_base")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def count_tokens(s: str) -> int:
    return len(ENC.encode(s))

def split_markdown_sections(text: str) -> List[Dict[str, Any]]:
    """
    Split markdown into sections by headings.
    Returns list of {level, header, content}
    """
    lines = text.splitlines()
    sections: List[Dict[str, Any]] = []
    current = {"level": 0, "header": "", "content": []}

    for ln in lines:
        m = HEADING_RE.match(ln.strip())
        if m:
            # push previous
            if current["content"]:
                sections.append({
                    "level": current["level"],
                    "header": current["header"],
                    "content": "\n".join(current["content"]).strip()
                })
            # start new
            current = {"level": len(m.group(1)), "header": m.group(2).strip(), "content": []}
        else:
            current["content"].append(ln)
    # last
    if current["content"]:
        sections.append({
            "level": current["level"],
            "header": current["header"],
            "content": "\n".join(current["content"]).strip()
        })
    # If file had no headings, wrap as a single section
    if not sections:
        return [{"level": 1, "header": "", "content": text.strip()}]
    return sections

def smart_paragraphs(s: str) -> List[str]:
    # Split on blank lines; keep small lines together
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    return paras

def chunk_paragraphs(
    paragraphs: List[str],
    max_tokens: int = 900,
    overlap_tokens: int = 150
) -> List[str]:
    """
    Greedy pack paragraphs into chunks by token budget.
    Uses token overlap between adjacent chunks for retrieval robustness.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for p in paragraphs:
        ptoks = count_tokens(p)
        if cur and (cur_tokens + ptoks > max_tokens):
            # finalize current
            chunks.append("\n\n".join(cur).strip())
            # build overlap context
            if overlap_tokens > 0:
                # take trailing text from current as overlap seed
                overlap_text = tail_text("\n\n".join(cur), overlap_tokens)
                cur = [overlap_text] if overlap_text else []
                cur_tokens = count_tokens("\n\n".join(cur)) if cur else 0
            else:
                cur, cur_tokens = [], 0
        # add paragraph
        cur.append(p)
        cur_tokens += ptoks

    if cur:
        chunks.append("\n\n".join(cur).strip())

    return [c for c in chunks if c]

def tail_text(text: str, approx_tokens: int) -> str:
    """Take tail of text containing ~approx_tokens using token units."""
    ids = ENC.encode(text)
    if len(ids) <= approx_tokens:
        return text
    tail_ids = ids[-approx_tokens:]
    return ENC.decode(tail_ids)

def markdown_to_chunks(
    md_text: str,
    max_tokens: int = 900,
    overlap_tokens: int = 150
) -> List[Dict[str, Any]]:
    """
    Convert a markdown document into chunks while keeping heading context.
    Returns list of {chunk_text, header_path}
    """
    sections = split_markdown_sections(md_text)
    result: List[Dict[str, Any]] = []
    header_stack: List[Tuple[int, str]] = []  # (level, name)

    for sec in sections:
        lvl, head, content = sec["level"], sec["header"], sec["content"]
        # adjust header stack
        if lvl > 0:
            while header_stack and header_stack[-1][0] >= lvl:
                header_stack.pop()
            if head:
                header_stack.append((lvl, head))
        header_path = " > ".join(h for _, h in header_stack if h)

        paragraphs = smart_paragraphs(content)
        chunk_texts = chunk_paragraphs(paragraphs, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for ct in chunk_texts:
            result.append({
                "chunk_text": ct,
                "header_path": header_path
            })
    # If nothing produced (e.g., empty), fallback
    if not result and md_text.strip():
        result = [{"chunk_text": md_text.strip(), "header_path": ""}]
    return result
