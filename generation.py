"""RAG generation step — Claude answers a question from retrieved birds.

This closes the retrieval → answer loop. Given a user query and a list of
matches (from any of the search_* helpers in query.py), we build a small
context window of bird-article excerpts and ask Claude to answer.

Why this matters for the demo: it lets the audience see retrieval feeding an
LLM — good birds in, useful answer out. The prompt asks Claude to do its best
with whatever birds the search returned, so the answer tracks retrieval
quality rather than refusing on borderline matches.

The eval matrix continues to score retrieval (span / doc recall) only.
Generation is a visual demo layer, not a tracked metric.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv
load_dotenv()

import anthropic


# Same Claude family the agentic router uses; override via env if needed.
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-sonnet-5")


SYSTEM_PROMPT = """\
You are a helpful birding assistant. A search system has returned a handful of
bird-article excerpts for the user's question. Using those returned birds, do
your best to answer the question: name the bird(s) that fit and quote facts
from the excerpts. If the excerpts only partially cover the question, answer
with what they do support rather than refusing — work with the birds you were
given. Only say the context can't help if none of the birds are relevant at
all. Don't add bird knowledge from outside the excerpts.

Keep answers to 3-5 sentences. No preamble.
"""


def _bird_name(m) -> str:
    """Pretty bird name. Prefer the explicit field, fall back to the slug."""
    name = m.get("bird_name") if hasattr(m, "get") else None
    if name:
        return name
    return m._id.replace("_", " ").replace("-", " ").title()


def build_context(matches: list, max_chars: int = 6000) -> str:
    """Build a context block from retrieved matches. Each entry includes the
    bird's name and as much of its article content as fits in the budget.

    The body is whatever the helper put in ``m.get("body")``:
    - For text/visual/FTS helpers, that's the full article body.
    - For text_semantic / search_within, it's the concatenated top chunks
      (richer for the LLM than a single chunk).
    """
    blocks = []
    used = 0
    for m in matches:
        name = _bird_name(m)
        intro = (m.get("intro") if hasattr(m, "get") else "") or ""
        body = (m.get("body") if hasattr(m, "get") else "") or ""
        # Cap individual document size so one bird doesn't crowd out others
        body = body[:2500]
        block = f"=== {name} ===\n{intro}\n\n{body}".strip()
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


@dataclass
class GenerationResult:
    answer: str
    elapsed_ms: float
    context_chars: int
    model: str


def _user_message(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate(
    query: str, matches: list, *, stream: bool = False
) -> GenerationResult | Iterator[str]:
    """Generate an answer from retrieved matches.

    ``stream=False`` (default): returns a ``GenerationResult`` with the full
    text and timing — useful for tests and eval.

    ``stream=True``: returns an iterator of text chunks. The notebook uses
    this with ``mo.output.replace`` for a streaming live-demo feel.
    """
    context = build_context(matches)
    client = anthropic.Anthropic()
    user = _user_message(query, context)

    if stream:
        def _stream():
            try:
                with client.messages.stream(
                    model=GEN_MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user}],
                ) as s:
                    for text in s.text_stream:
                        yield text
            except Exception as e:
                yield f"[generation error: {e}]"
        return _stream()

    t0 = time.perf_counter()
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    answer = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return GenerationResult(
        answer=answer,
        elapsed_ms=elapsed_ms,
        context_chars=len(context),
        model=GEN_MODEL,
    )
