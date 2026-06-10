"""Smoke tests for generation.py — Claude answers from retrieved matches.

We feed hand-built matches so these test GENERATION, not retrieval. Requires
ANTHROPIC_API_KEY (Claude does the generation); skips if absent.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or None
pytestmark = pytest.mark.skipif(
    ANTHROPIC_KEY is None, reason="ANTHROPIC_API_KEY not set"
)


def _match(_id: str, **fields):
    """A fake match object supporting both ``m._id`` and ``m.get(field)``."""
    return SimpleNamespace(_id=_id, get=lambda k, d=None, _f=fields: _f.get(k, d))


def test_build_context_assembles_blocks():
    from generation import build_context

    m1 = _match(
        "Northern_mockingbird",
        bird_name="Northern mockingbird",
        intro="The mockingbird mimics other birds.",
        body="It is known for vocal mimicry.",
    )
    m2 = _match(
        "European_starling",
        bird_name="European starling",
        intro="Starlings imitate sounds.",
        body="They have a wide repertoire.",
    )
    ctx = build_context([m1, m2])
    assert "Northern mockingbird" in ctx
    assert "European starling" in ctx
    assert "===" in ctx


def test_generate_uses_the_provided_context():
    """Good context → the answer names the mimic from the excerpts."""
    from generation import generate

    matches = [
        _match(
            "Northern_mockingbird",
            bird_name="Northern mockingbird",
            intro="The northern mockingbird is known for vocal mimicry.",
            body=(
                "The northern mockingbird imitates the calls of other birds, "
                "insects, and amphibians, often repeating each phrase several times."
            ),
        ),
    ]
    out = generate("Which bird is best known for imitating other sounds?", matches)
    answer = out.answer.lower()
    assert len(out.answer) > 20, f"answer too short: {out.answer!r}"
    assert "mockingbird" in answer, f"expected the mockingbird in: {out.answer!r}"
    print(f"\n[good context]\n{out.answer}\n")


def test_generate_with_irrelevant_context():
    """Off-topic context → still runs and returns text (a human can read that the
    answer reflects the wrong birds). We only assert it produced something."""
    from generation import generate

    matches = [
        _match(
            "California_gull",
            bird_name="California gull",
            intro="The California gull is a medium-sized gull.",
            body="The California gull famously ate the Mormon crickets that plagued Utah settlers.",
        ),
    ]
    out = generate("Which bird is best known for imitating other sounds?", matches)
    assert len(out.answer) > 0
    print(f"\n[irrelevant context]\n{out.answer}\n")
