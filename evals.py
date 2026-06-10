"""Eval primitives for the bird-search teaching demo.

Hand-authored ground truth in ``ground_truth.json`` drives two retrieval
metrics:

- ``doc_recall_at_k``: fraction of the query's expected bird slugs that
  appear in the top-k matches. Bird-level — "did we surface the right
  bird(s)?"
- ``recall_at_k`` (span recall): fraction of the query's expected text spans
  (literal substrings or regex-ish ``match_pattern``s) found in the
  retrieved articles' body / intro fields. Passage-level — "did we
  retrieve articles containing the right content?"

Both are simple, transparent, and run in the notebook in seconds. No
LLM judge, no embedding-similarity fudge — just substring / regex
matching against normalized text. A query passes if it meets its
``minimum_doc_recall`` (when set) or ``recall_at_k >= 0.5`` otherwise.

The notebook imports these to build its recap scorecard.
"""

from __future__ import annotations

import json
import re
from typing import Any


# -----------------------------------------------------------------------------
# Text normalization + span matching
# -----------------------------------------------------------------------------

def normalize(text: str) -> str:
    """lowercase + strip punctuation + collapse whitespace.

    Substring checks run against this normalized form so "Mormon-crickets,"
    matches "mormon crickets" without false negatives from punctuation.
    """
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def span_found_in_match(span: dict, match: Any) -> bool:
    """True if the span's text/pattern appears in ``match``'s ``field``.

    ``span`` shape (per ground_truth.json):
        {"doc_id": "*"|"<slug>", "field": "body"|"intro"|...,
         "text": "<literal substring>"}                    # literal
        OR
        {"doc_id": ..., "field": ..., "match_pattern": "<re|alternation>"}   # regex
    """
    field = span.get("field", "body")
    field_text = match.get(field) or ""
    if not field_text:
        return False
    if "text" in span:
        return normalize(span["text"]) in normalize(field_text)
    if "match_pattern" in span:
        return bool(re.search(span["match_pattern"], field_text, re.IGNORECASE))
    return False


def recall_at_k(query_gt: dict, matches: list, k: int = 5) -> float:
    """Fraction of ``required_spans`` satisfied by the top-k matches.

    For each span:
    - If ``doc_id`` is ``"*"`` (or missing), the span is satisfied by ANY
      match in top-k whose ``field`` contains the text/pattern.
    - Otherwise, the match with that specific ``_id`` must be in top-k AND
      contain the span — pins the span to a specific bird.
    """
    spans = query_gt.get("required_spans", [])
    if not spans:
        return 1.0
    top_k = matches[:k]
    hits = 0
    for span in spans:
        doc_id = span.get("doc_id")
        if doc_id in ("*", None):
            if any(span_found_in_match(span, m) for m in top_k):
                hits += 1
        else:
            for m in top_k:
                if m._id == doc_id and span_found_in_match(span, m):
                    hits += 1
                    break
    return hits / len(spans)


def doc_recall_at_k(query_gt: dict, matches: list, k: int = 5) -> float:
    """Fraction of ``required_doc_ids`` that appear in top-k by _id."""
    required = set(query_gt.get("required_doc_ids", []))
    if not required:
        return 1.0
    top_k_ids = {m._id for m in matches[:k]}
    found = required & top_k_ids
    return len(found) / len(required)


def passes(query_gt: dict, matches: list, k: int = 5) -> bool:
    """Pass = meets minimum_doc_recall (when set) OR recall_at_k >= 0.5.

    The two flavors exist because:
    - Q1-Q4 have a small set of "right answer" birds where surfacing 1+ is
      effectively passing — minimum_doc_recall expresses that floor.
    - Q5/Q6 want a *set* of plausible candidates; same model.
    """
    if "minimum_doc_recall" in query_gt:
        required = query_gt.get("required_doc_ids", [])
        top_k_ids = {m._id for m in matches[:k]}
        return len(set(required) & top_k_ids) >= query_gt["minimum_doc_recall"]
    return recall_at_k(query_gt, matches, k) >= 0.5


def load_ground_truth(path: str = "ground_truth.json") -> dict:
    with open(path) as f:
        return json.load(f)
