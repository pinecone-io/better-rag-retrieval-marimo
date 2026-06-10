"""Bird Search — shared Pinecone / Gemini plumbing.

The notebook authors each retrieval technique inline, so readers see the real
``documents.search(...)`` call in the cell itself. This module holds only the
shared pieces those cells import: the Pinecone client, the Gemini text-embedding
helper, the two index names, and a tiny match wrapper.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

from google import genai
from google.genai import types as genai_types


# The two indexes the demo queries (built once by build_index.py /
# build_text_index.py — see the README).
INDEX = "bird-search-fts"
TEXT_INDEX = "bird-search-text-dense"
NAMESPACE = "birds"

GEMINI_MODEL = "gemini-embedding-2"
GEMINI_EMBED_DIMENSIONS = 768


pc = Pinecone(source_tag="pinecone:bird_search_retrieval_rag")
gem = genai.Client()  # reads GOOGLE_API_KEY


_EMBED_CONFIG = genai_types.EmbedContentConfig(
    output_dimensionality=GEMINI_EMBED_DIMENSIONS,
)


def embed_text(text: str) -> list[float]:
    """Embed a text query into Gemini-2's multimodal space, truncated to
    ``GEMINI_EMBED_DIMENSIONS``. Text and image embeddings share this space,
    so a text query can be scored directly against stored image vectors.
    """
    resp = gem.models.embed_content(
        model=GEMINI_MODEL, contents=text, config=_EMBED_CONFIG
    )
    return list(resp.embeddings[0].values)


class _BirdMatch:
    """A tiny stand-in for an SDK match object, used when chunk-level hits are
    aggregated back to one row per bird. Supports the surface the notebook and
    eval code touch: ``_id`` / ``_score`` / ``score`` (attrs), ``get(field)``,
    and ``to_dict()``.
    """

    __slots__ = ("_id", "_score", "score", "_fields")

    def __init__(self, _id: str, score: float, fields: dict):
        self._id = _id
        self._score = score
        self.score = score  # mirror; some call sites read `m.score`
        self._fields = fields

    def get(self, field: str, default: Any = None) -> Any:
        return self._fields.get(field, default)

    def to_dict(self) -> dict:
        return {"_id": self._id, "_score": self._score, **self._fields}

    def __repr__(self) -> str:
        return f"_BirdMatch(_id={self._id!r}, _score={self._score:.4f})"
