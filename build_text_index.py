"""Bird Search v2 — sister text-dense index ingestion.

Builds a *second* Pinecone preview FTS index over the bird corpus, with one
dense vector field (``text_embedding``) populated by Gemini Embedding 2 over
chunked article text. This sister index complements the main ``bird-search-fts``
index (which carries the image embedding); together they enable:

- ``search_text_semantic(q, top_k)`` — dense semantic on text
- ``search_within(filter_terms, image_q, semantic_q, top_k)`` — coarse-to-fine
  retrieval (FTS / image gate on main index → dense-text refine on this index)

Why a sister index? Pinecone FTS in 2026-01.alpha enforces "at most one
dense_vector field per index" (verified empirically). Adding a text dense
vector therefore requires a second index.

Why chunked? Wikipedia bird articles range from 50 words (rare species) to
13k+ words (red-winged blackbird). Whole-article embedding either (a) clips
content past Gemini's input limit or (b) blurs many topics into one vector,
hurting precision. Paragraph-aware chunks (~800 words) align with semantic
units and let dense retrieval surface the *passage* answering a how/why query.

Usage:
    python build_text_index.py [--create-only] [--sample N] [--recreate]
                               [--clear-cache] [--data-dir PATH]

Defaults:
    --sample 0             (0 = full corpus; pass a small N for smoke tests)
    --data-dir             $BIRD_DATA_DIR or ./parsed_birds

Embeddings are cached at text-embeddings-cache.jsonl (one row per chunk,
tagged with model + dim + chunker version). Re-running resumes; pass
--clear-cache to force re-embedding after a config change.

Env:
    PINECONE_API_KEY       required
    GOOGLE_API_KEY         required
    BIRD_DATA_DIR          overrides default parsed_birds path
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

from pinecone import Pinecone
from pinecone.preview import SchemaBuilder

from google import genai
from google.genai import types as genai_types


# -----------------------------------------------------------------------------
# Constants — kept in sync with build_index.py where shared
# -----------------------------------------------------------------------------

TEXT_INDEX = "bird-search-text-dense"
NAMESPACE = "birds"

GEMINI_MODEL = "gemini-embedding-2"
GEMINI_EMBED_DIMENSIONS = 768

# Chunker parameters. Paragraph-aware packing: greedily add paragraphs to a
# chunk until target_words is reached; never exceed max_words in a single
# chunk (split that paragraph by sentence-ish breaks if needed). Each new
# chunk starts by re-including the previous chunk's LAST paragraph as
# overlap — preserves cross-paragraph references.
CHUNKER_VERSION = "v2-paragraph-800w-1paraover"
# Gemini-embedding-2 supports ~2048 input tokens. Target ~60% of that
# (~800 words ≈ ~1200 tokens) so each chunk genuinely uses the model's
# capacity; cap at ~90% (~1200 words ≈ ~1800 tokens). This keeps
# paragraphs intact in the common case (median article ~640 words →
# 1 chunk; major-species articles 3-10k words → 4-14 chunks).
CHUNK_TARGET_WORDS = 800
CHUNK_MAX_WORDS = 1200
CHUNK_OVERLAP_PARAS = 1


_DEFAULT_DATA_DIR_PATH = pathlib.Path(__file__).resolve().parent / "parsed_birds"
DEFAULT_DATA_DIR = os.environ.get("BIRD_DATA_DIR", str(_DEFAULT_DATA_DIR_PATH))

EMBED_CACHE_PATH = pathlib.Path(__file__).resolve().parent / "text-embeddings-cache.jsonl"

EMBED_CONCURRENCY = 8
EMBED_MAX_RETRIES = 5
EMBED_BASE_BACKOFF_S = 2.0


pc = Pinecone(source_tag="pinecone:bird_search_example:text_dense")
gem = genai.Client()


_EMBED_CONFIG = genai_types.EmbedContentConfig(
    output_dimensionality=GEMINI_EMBED_DIMENSIONS,
)


def embed_text(text: str) -> list[float]:
    """Embed a string into Gemini-2's multimodal space at 768 dims."""
    resp = gem.models.embed_content(
        model=GEMINI_MODEL, contents=text, config=_EMBED_CONFIG
    )
    return list(resp.embeddings[0].values)


# -----------------------------------------------------------------------------
# Chunker — paragraph-aware packing with one-paragraph overlap
# -----------------------------------------------------------------------------

def chunk_article(
    text: str,
    target_words: int = CHUNK_TARGET_WORDS,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_paras: int = CHUNK_OVERLAP_PARAS,
) -> list[str]:
    """Pack paragraphs into ~target_words chunks, capped at max_words.

    Each chunk after the first begins by re-including the last
    `overlap_paras` paragraph(s) of the previous chunk for context
    continuity. A paragraph that alone exceeds max_words is split on
    sentence-ish breaks ('. ', '! ', '? ') into sub-paragraphs.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []

    # Step 1: split any single paragraph longer than max_words by sentences
    safe_paras: list[str] = []
    for p in paras:
        if len(p.split()) <= max_words:
            safe_paras.append(p)
            continue
        # Sentence-ish split — keep delimiters, group up to max_words
        import re
        sents = re.split(r"(?<=[.!?])\s+", p)
        buf: list[str] = []
        bw = 0
        for s in sents:
            sw = len(s.split())
            if bw + sw > max_words and buf:
                safe_paras.append(" ".join(buf))
                buf, bw = [s], sw
            else:
                buf.append(s)
                bw += sw
        if buf:
            safe_paras.append(" ".join(buf))

    # Step 2: pack into chunks
    chunks: list[str] = []
    i = 0
    while i < len(safe_paras):
        chunk_paras: list[str] = []
        words = 0
        j = i
        while j < len(safe_paras):
            pw = len(safe_paras[j].split())
            if chunk_paras and words + pw > max_words:
                break
            chunk_paras.append(safe_paras[j])
            words += pw
            j += 1
            if words >= target_words:
                break
        chunks.append("\n\n".join(chunk_paras))
        if j >= len(safe_paras):
            break
        # Advance by chunk-size minus overlap; never less than +1 to avoid infinite loop
        i = max(j - overlap_paras, i + 1)

    return chunks


# -----------------------------------------------------------------------------
# Metadata loading (mirrors build_index.py's filter)
# -----------------------------------------------------------------------------

def load_metadata(data_dir: pathlib.Path) -> dict[str, Any]:
    meta_path = data_dir / "parsing_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"parsing_metadata.json not found at {meta_path}. "
            "Set BIRD_DATA_DIR or pass --data-dir."
        )
    return json.loads(meta_path.read_text())


def filter_usable_slugs(meta: dict[str, Any], data_dir: pathlib.Path) -> list[str]:
    """Return slugs whose text file exists. (Images aren't required for the
    text-dense index — but we keep the same filter so the two indexes cover
    the same bird set.)
    """
    usable: list[str] = []
    missing_text = 0
    no_images = 0
    missing_image = 0
    for slug, entry in meta.items():
        text_path = data_dir / "text" / entry.get("text_file", "")
        if not text_path.exists():
            missing_text += 1
            continue
        images = entry.get("images") or []
        if not images:
            no_images += 1
            continue
        img_path = data_dir / "images" / images[0]["local_path"]
        if not img_path.exists():
            missing_image += 1
            continue
        usable.append(slug)
    total = len(meta)
    print(
        f"Filtered corpus: {len(usable):,} usable / {total:,} total "
        f"({missing_text} no-text, {no_images} no-image-meta, {missing_image} missing-image)"
    )
    return usable


def split_intro_body(text: str) -> tuple[str, str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return "", ""
    return paragraphs[0], "\n\n".join(paragraphs[1:])


# -----------------------------------------------------------------------------
# Embedding cache + concurrent fill (mirrors build_index.py pattern)
# -----------------------------------------------------------------------------

def _embed_with_retry(text: str) -> list[float]:
    last_exc: Exception | None = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            return embed_text(text)
        except Exception as exc:
            last_exc = exc
            if attempt == EMBED_MAX_RETRIES - 1:
                break
            sleep_s = EMBED_BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"  embed retry in {sleep_s:.1f}s (attempt {attempt + 1}/"
                f"{EMBED_MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def load_embedding_cache(path: pathlib.Path) -> dict[str, list[float]]:
    """Read JSONL cache. Key = `<slug>#<chunk_idx>`. Stale rows (different
    model / dim / chunker version) are silently skipped.
    """
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    stale = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stale += 1
                continue
            if (
                row.get("model") != GEMINI_MODEL
                or row.get("dim") != GEMINI_EMBED_DIMENSIONS
                or row.get("chunker") != CHUNKER_VERSION
            ):
                stale += 1
                continue
            cache[row["_id"]] = row["embedding"]
    msg = f"Embedding cache: {len(cache):,} usable rows from {path.name}"
    if stale:
        msg += f" ({stale} stale skipped)"
    print(msg)
    return cache


def append_embedding_cache(
    path: pathlib.Path, _id: str, text: str, embedding: list[float]
) -> None:
    row = {
        "_id": _id,
        "model": GEMINI_MODEL,
        "dim": GEMINI_EMBED_DIMENSIONS,
        "chunker": CHUNKER_VERSION,
        "embedding": embedding,
    }
    with path.open("a") as f:
        f.write(json.dumps(row))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


# -----------------------------------------------------------------------------
# Document building
# -----------------------------------------------------------------------------

def build_chunks_for_bird(
    slug: str, entry: dict[str, Any], data_dir: pathlib.Path
) -> list[dict[str, Any]]:
    """Return [{_id, slug, bird_name, chunk_idx, chunk_text}] for one bird.
    `text_embedding` is filled later from the cache.
    """
    text_path = data_dir / "text" / entry["text_file"]
    text = text_path.read_text(encoding="utf-8")
    intro, body = split_intro_body(text)
    # Embed intro+body as one stream of paragraphs (intro is just paragraph 0)
    full = (intro + "\n\n" + body).strip() if intro else body
    pieces = chunk_article(full)
    if not pieces:
        return []
    bird_name = slug.replace("_", " ")
    return [
        {
            "_id": f"{slug}#{i}",
            "slug": slug,
            "bird_name": bird_name,
            "chunk_text": piece,
        }
        for i, piece in enumerate(pieces)
    ]


# -----------------------------------------------------------------------------
# Schema + index management
# -----------------------------------------------------------------------------

def build_schema(embed_dim: int):
    # Only FTS-enabled string fields and vector fields are declared in the
    # schema. Plain metadata (slug, bird_name) is passed at upsert time and
    # automatically indexed for filter operators ($in, $eq, etc).
    # chunk_idx is encoded into _id ("<slug>#<idx>") — recoverable at read time.
    return (
        SchemaBuilder()
        .add_string_field(
            "chunk_text",
            full_text_search={"language": "en", "stemming": True},
        )
        .add_dense_vector_field(
            "text_embedding",
            dimension=embed_dim,
            metric="cosine",
        )
        .build()
    )


def ensure_index(embed_dim: int, recreate: bool) -> None:
    if pc.preview.indexes.exists(TEXT_INDEX):
        if recreate:
            print(f"Deleting existing index '{TEXT_INDEX}'...")
            pc.preview.indexes.delete(TEXT_INDEX)
            while pc.preview.indexes.exists(TEXT_INDEX):
                time.sleep(2)
        else:
            print(f"Index '{TEXT_INDEX}' already exists. (Pass --recreate to drop.)")
            return
    schema = build_schema(embed_dim)
    pc.preview.indexes.create(name=TEXT_INDEX, schema=schema)
    print(f"Created index '{TEXT_INDEX}'.")


def wait_until_ready(timeout_s: int = 300) -> None:
    print(f"Waiting for index '{TEXT_INDEX}' to become ready...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = pc.preview.indexes.describe(TEXT_INDEX)
        if info.status.ready:
            print(f"Index is ready (state={info.status.state!r}).")
            return
        print(f"  {info.status.state} — sleeping 5 s...")
        time.sleep(5)
    raise TimeoutError(f"Index {TEXT_INDEX} not ready after {timeout_s}s")


def wait_until_searchable(idx, timeout_s: int = 300) -> None:
    print("Waiting for documents to be indexed...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = idx.documents.search(
            namespace=NAMESPACE,
            top_k=1,
            score_by=[{"type": "text", "field": "chunk_text", "query": "bird"}],
            include_fields=[],
        )
        if resp.matches:
            print("  Data is searchable.")
            return
        time.sleep(5)
        print("  Not yet indexed, retrying...", flush=True)
    print("WARNING: Documents may not be fully indexed after timeout.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--create-only", action="store_true",
                        help="Create the index and exit (no embed/upsert).")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of birds to ingest (default 0 = full).")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the index before ingest.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete text-embeddings-cache.jsonl before ingest.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Path to parsed_birds (default $BIRD_DATA_DIR or {DEFAULT_DATA_DIR}).")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
    print(f"Using data dir: {data_dir}")

    meta = load_metadata(data_dir)

    if args.clear_cache and EMBED_CACHE_PATH.exists():
        print(f"Clearing embedding cache at {EMBED_CACHE_PATH}")
        EMBED_CACHE_PATH.unlink()

    if args.create_only:
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=args.recreate)
        wait_until_ready()
        print(f"Index '{TEXT_INDEX}' is Ready. Re-run without --create-only to ingest.")
        return

    if args.recreate:
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=True)
        wait_until_ready()
    elif not pc.preview.indexes.exists(TEXT_INDEX):
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=False)
        wait_until_ready()

    usable_slugs = filter_usable_slugs(meta, data_dir)
    if args.sample and args.sample > 0:
        slugs = usable_slugs[: args.sample]
    else:
        slugs = usable_slugs
    print(f"Preparing to ingest {len(slugs)} / {len(usable_slugs)} birds.")

    # ---- Chunk every bird upfront -------------------------------------------
    all_chunks: list[dict[str, Any]] = []
    for slug in tqdm(slugs, desc="Chunking"):
        all_chunks.extend(build_chunks_for_bird(slug, meta[slug], data_dir))
    print(f"Built {len(all_chunks):,} chunks across {len(slugs)} birds "
          f"(avg {len(all_chunks)/max(len(slugs),1):.1f} per bird).")

    # ---- Embedding phase (concurrent, resumable) ----------------------------
    cache = load_embedding_cache(EMBED_CACHE_PATH)
    missing = [c for c in all_chunks if c["_id"] not in cache]

    if missing:
        print(f"Embedding {len(missing):,} chunks (concurrency={EMBED_CONCURRENCY})...")
        failures: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
            futs = {
                pool.submit(_embed_with_retry, c["chunk_text"]): c
                for c in missing
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Embedding"):
                c = futs[fut]
                try:
                    emb = fut.result()
                except Exception as exc:
                    failures.append((c["_id"], str(exc)))
                    continue
                cache[c["_id"]] = emb
                append_embedding_cache(EMBED_CACHE_PATH, c["_id"], c["chunk_text"], emb)
        if failures:
            print(f"\nEmbed failures: {len(failures)} / {len(missing)}", file=sys.stderr)
            for cid, err in failures[:5]:
                print(f"  {cid}: {err}", file=sys.stderr)
    else:
        print("All chunks already cached.")

    # ---- Build upsert payload ----------------------------------------------
    docs: list[dict[str, Any]] = []
    skipped = 0
    for c in all_chunks:
        if c["_id"] not in cache:
            skipped += 1
            continue
        docs.append({**c, "text_embedding": cache[c["_id"]]})
    print(f"Built {len(docs):,} documents ({skipped} skipped).")
    if not docs:
        print("No documents to upsert — exiting.")
        return

    # ---- Upsert ------------------------------------------------------------
    idx = pc.preview.index(name=TEXT_INDEX)
    # 50 docs per batch with a few concurrent workers — same shape as the
    # image index; plenty for the ~3,500 chunks this corpus produces.
    result = idx.documents.batch_upsert(
        namespace=NAMESPACE,
        documents=docs,
        batch_size=50,
        max_workers=4,
        show_progress=True,
    )
    print(f"\nUploaded {result.successful_item_count:,} / {result.total_item_count:,} documents")
    if getattr(result, "has_errors", False):
        print(f"Failed batches: {result.failed_batch_count} / {result.total_batch_count}",
              file=sys.stderr)
        for err in list(getattr(result, "errors", []))[:3]:
            msg = getattr(err, "error_message", None) or str(getattr(err, "error", err))
            sample_id = err.items[0].get("_id") if getattr(err, "items", None) else "?"
            print(
                f"  batch #{getattr(err, 'batch_index', '?')} "
                f"({len(err.items)} items, first _id={sample_id!r}): {msg}",
                file=sys.stderr,
            )
        if result.successful_item_count == 0:
            raise SystemExit("All upserts failed — aborting before search-poll.")

    wait_until_searchable(idx)


if __name__ == "__main__":
    main()
