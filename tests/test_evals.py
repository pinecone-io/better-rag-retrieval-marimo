"""Smoke tests for evals primitives — no Pinecone calls, pure functions."""

from types import SimpleNamespace

from evals import (
    normalize,
    span_found_in_match,
    recall_at_k,
    doc_recall_at_k,
    passes,
)


def _match(_id: str, **fields):
    """Build a fake match object that supports both `m._id` and `m.get(f)`."""
    return SimpleNamespace(_id=_id, get=lambda k, d=None: fields.get(k, d))


def test_normalize_collapses_whitespace_and_punct():
    assert normalize("  Mormon-crickets,  PLAGUE!  ") == "mormon crickets plague"
    assert normalize(None) == ""
    assert normalize("") == ""


def test_span_found_with_literal_text():
    m = _match("california_gull", body="settlers dealing with a plague of Mormon crickets near Salt Lake")
    span = {"doc_id": "california_gull", "field": "body", "text": "Mormon crickets"}
    assert span_found_in_match(span, m) is True
    # Punctuation around the substring shouldn't matter
    m2 = _match("x", body="The Mormon-crickets infestation...")
    assert span_found_in_match(span, m2) is True


def test_span_found_with_match_pattern():
    m = _match("northern_mockingbird", body="able to imitate the calls of other birds")
    span = {"doc_id": "*", "field": "body", "match_pattern": "imitat|mimic|vocal mimicry"}
    assert span_found_in_match(span, m) is True
    span_miss = {"doc_id": "*", "field": "body", "match_pattern": "carotenoid"}
    assert span_found_in_match(span_miss, m) is False


def test_span_missing_field_returns_false():
    m = _match("x", body="")  # explicitly empty
    span = {"doc_id": "*", "field": "body", "text": "anything"}
    assert span_found_in_match(span, m) is False


def test_recall_at_k_partial():
    gt = {
        "required_spans": [
            {"doc_id": "*", "field": "body", "match_pattern": "imitat|mimic"},
            {"doc_id": "*", "field": "body", "match_pattern": "northern"},
            {"doc_id": "*", "field": "body", "match_pattern": "nonexistentword12345"},
        ]
    }
    matches = [
        _match("nm", body="The Northern mockingbird can imitate other birds"),
    ]
    # 2 of 3 spans hit
    assert recall_at_k(gt, matches, k=5) == 2 / 3


def test_recall_at_k_doc_pinned():
    gt = {
        "required_spans": [
            {"doc_id": "california_gull", "field": "body", "text": "Mormon crickets"},
        ]
    }
    # Wrong doc has the text — should NOT count
    matches = [_match("other_bird", body="Mormon crickets are also eaten by gulls")]
    assert recall_at_k(gt, matches, k=5) == 0.0
    # Right doc has the text — should count
    matches2 = [_match("california_gull", body="ate Mormon crickets and saved the crops")]
    assert recall_at_k(gt, matches2, k=5) == 1.0


def test_doc_recall_at_k():
    gt = {"required_doc_ids": ["a", "b", "c"]}
    assert doc_recall_at_k(gt, [_match("a"), _match("b")], k=5) == 2 / 3
    assert doc_recall_at_k(gt, [_match("x"), _match("y")], k=5) == 0.0
    assert doc_recall_at_k(gt, [_match("a"), _match("b"), _match("c")], k=5) == 1.0


def test_passes_with_minimum_doc_recall():
    gt = {
        "required_doc_ids": ["a", "b", "c", "d"],
        "minimum_doc_recall": 2,
    }
    assert passes(gt, [_match("a")], k=5) is False
    assert passes(gt, [_match("a"), _match("b")], k=5) is True
    assert passes(gt, [_match("a"), _match("b"), _match("c"), _match("d")], k=5) is True


def test_passes_with_span_recall_default_threshold():
    gt = {
        "required_spans": [
            {"doc_id": "*", "field": "body", "text": "x"},
            {"doc_id": "*", "field": "body", "text": "y"},
        ]
    }
    # 0/2 spans → fail
    assert passes(gt, [_match("m", body="nothing")], k=5) is False
    # 1/2 spans → 0.5 == threshold → pass
    assert passes(gt, [_match("m", body="x is here")], k=5) is True
