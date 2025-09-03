# utils/prompts.py
from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful retail knowledge assistant.
Answer ONLY using the provided context chunks.
Start with a single crisp line that answers the question directly (e.g., “Return window: 30 days.”).
Then add 1–3 short supporting bullets if useful.
Always include citations like [Title §Header].
If the answer is not in the context, say: “I don’t know based on the provided documents.”

Always include citations like [Title §Header] for each claim you make.
Be concise and specific."""

USER_PROMPT_TEMPLATE = """Question:
{question}

Context chunks:
{context}

Instructions:
- Use only the information in the context.
- Cite sources inline using: [Title §Header]
- If insufficient information, say you don't know based on the documents.
- Return a crisp answer first, then (if helpful) bullet points."""

def format_context(chunks):
    # chunks: list of dicts with keys: title, header_path, text
    lines = []
    for i, ch in enumerate(chunks, start=1):
        hdr = f" §{ch['header_path']}" if ch.get("header_path") else ""
        lines.append(f"[{i}] {ch['title']}{hdr}\n{ch['text']}\n")
    return "\n".join(lines)
