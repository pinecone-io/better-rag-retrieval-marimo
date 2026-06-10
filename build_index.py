"""Bird Search v2 — ingestion pipeline.

Builds a Pinecone preview FTS index over a sample of the parsed Wikipedia bird
corpus, with three full-text fields (bird_name, intro, body) and one dense
vector field (image_embedding) populated from Gemini Embedding 2 over each
bird's primary image.

Usage:
    python build_index.py [--create-only] [--sample N] [--recreate]
                          [--clear-cache] [--data-dir PATH]

Defaults:
    --sample 50            (pass a large N or 0 to ingest the full corpus)
    --data-dir             $BIRD_DATA_DIR or ./parsed_birds (in-repo)

Embeddings are cached at embeddings-cache.jsonl (one row per bird, tagged
with model + dim). Re-running resumes where a previous run stopped; pass
--clear-cache to force re-embedding after a model or dim change.

Env:
    PINECONE_API_KEY       required (read via Pinecone() default)
    GOOGLE_API_KEY         required (read via genai.Client() default)
    BIRD_DATA_DIR          overrides the default parsed_birds path
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

from PIL import Image
from tqdm import tqdm

from pinecone import Pinecone
from pinecone.preview import SchemaBuilder

from google import genai
from google.genai import types as genai_types


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

INDEX = "bird-search-fts"
NAMESPACE = "birds"

GEMINI_MODEL = "gemini-embedding-2"

GEMINI_EMBED_DIMENSIONS = 768


_DEFAULT_DATA_DIR_PATH = pathlib.Path(__file__).resolve().parent / "parsed_birds"
DEFAULT_DATA_DIR = os.environ.get("BIRD_DATA_DIR", str(_DEFAULT_DATA_DIR_PATH))

# -----------------------------------------------------------------------------
# Embedding cache + concurrency
# -----------------------------------------------------------------------------

# Resumable cache of Gemini image embeddings — one JSONL line per bird. If
# the ingest is interrupted (Ctrl+C, rate limits, crash), re-running picks up
# only the birds not already present. Each line is tagged with the model ID
# and output dimension so a config change silently invalidates old rows on
# load rather than reusing stale vectors against a new schema.
EMBED_CACHE_PATH = pathlib.Path(__file__).resolve().parent / "embeddings-cache.jsonl"

# Will need to change or adjust this if using without paid tier
EMBED_CONCURRENCY = 8

# Exponential backoff bounds on a single embed call.
# covers typical transient failures (429s, 5xxs) without stalling the whole
# ingest on a permanent outage.
EMBED_MAX_RETRIES = 5
EMBED_BASE_BACKOFF_S = 2.0


# -----------------------------------------------------------------------------
# Client handles — lazy at module scope so imports are cheap / testable.
# -----------------------------------------------------------------------------

pc = Pinecone(source_tag="pinecone:bird_search_example")
gem = genai.Client()     # reads GOOGLE_API_KEY



_EMBED_CONFIG = genai_types.EmbedContentConfig(
    output_dimensionality=GEMINI_EMBED_DIMENSIONS,
)


def embed_image(pil_image: Image.Image) -> list[float]:
    """Embed a PIL image into Gemini-2's multimodal space, truncated to
    GEMINI_EMBED_DIMENSIONS."""
    resp = gem.models.embed_content(
        model=GEMINI_MODEL, contents=pil_image, config=_EMBED_CONFIG
    )
    return list(resp.embeddings[0].values)


# -----------------------------------------------------------------------------
# Bird loading
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
    """Return slugs for birds with both a readable text file and a primary
    image that exists on disk. Birds missing either are dropped upfront so
    that --sample N actually ingests N birds (not N-minus-skips).

    Prints a summary of what was filtered out and why.
    """
    usable: list[str] = []
    no_images_meta = 0
    missing_text = 0
    missing_image = 0
    for slug, entry in meta.items():
        text_path = data_dir / "text" / entry.get("text_file", "")
        if not text_path.exists():
            missing_text += 1
            continue
        images = entry.get("images") or []
        if not images:
            no_images_meta += 1
            continue
        img_path = data_dir / "images" / images[0]["local_path"]
        if not img_path.exists():
            missing_image += 1
            continue
        usable.append(slug)

    total = len(meta)
    dropped = total - len(usable)
    print(
        f"Filtered corpus: {len(usable):,} usable / {total:,} total "
        f"({dropped} dropped — {no_images_meta} no-image-in-metadata, "
        f"{missing_text} missing-text-file, {missing_image} missing-image-file)"
    )
    return usable


def split_intro_body(text: str) -> tuple[str, str]:
    """Split article text into (intro, body) on blank-line paragraph breaks.

    Intro = first non-empty paragraph
    Body  = remaining paragraphs joined with "\\n\\n".
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return "", ""
    return paragraphs[0], "\n\n".join(paragraphs[1:])


def embed_image_with_retry(pil_image: Image.Image) -> list[float]:
    """Gemini embed call with exponential backoff + jitter on transient errors.

    Raises the last exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            return embed_image(pil_image)
        except Exception as exc:
            last_exc = exc
            if attempt == EMBED_MAX_RETRIES - 1:
                break
            sleep_s = EMBED_BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"  embed retry in {sleep_s:.1f}s (attempt {attempt + 1}"
                f"/{EMBED_MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def load_embedding_cache(path: pathlib.Path) -> dict[str, list[float]]:
    """Read the JSONL cache and return ``{slug: embedding}`` for rows whose
    ``model`` and ``dim`` match the current config. Stale rows (different
    model or dim) are silently skipped so config changes don't poison results.
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
                # Either a stale row from an earlier model/dim, or a partial
                # last line from a hard crash mid-write. Skip either way.
                print(
                    f"  skipping unparseable cache line: {line[:80]!r}",
                    file=sys.stderr,
                )
                stale += 1
                continue
            if (
                row.get("model") != GEMINI_MODEL
                or row.get("dim") != GEMINI_EMBED_DIMENSIONS
            ):
                stale += 1
                continue
            cache[row["slug"]] = row["embedding"]
    msg = f"Embedding cache: {len(cache):,} usable rows from {path.name}"
    if stale:
        msg += f" ({stale} stale rows skipped — different model or dim)"
    print(msg)
    return cache


def append_embedding_cache(
    path: pathlib.Path, slug: str, embedding: list[float]
) -> None:
    """Append one JSONL row and fsync so a Ctrl+C mid-run preserves progress."""
    row = {
        "slug": slug,
        "model": GEMINI_MODEL,
        "dim": GEMINI_EMBED_DIMENSIONS,
        "embedding": embedding,
    }
    with path.open("a") as f:
        f.write(json.dumps(row))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _embed_one_bird(
    slug: str, entry: dict[str, Any], data_dir: pathlib.Path
) -> tuple[str, list[float] | None, str | None]:
    """Worker body: open the bird's primary image, embed with retry.

    Returns (slug, embedding or None, error_message or None). Image
    decoding and the Gemini call happen entirely inside the worker so
    concurrent threads don't contend on a single open file handle.
    """
    try:
        img_path = data_dir / "images" / entry["images"][0]["local_path"]
        with Image.open(img_path) as im:
            im.load()
            emb = embed_image_with_retry(im)
        return slug, emb, None
    except Exception as exc:
        return slug, None, str(exc)


def fill_embedding_cache(
    slugs: list[str],
    meta: dict[str, Any],
    data_dir: pathlib.Path,
    cache: dict[str, list[float]],
) -> None:
    """Embed any slugs not already in `cache`, using a thread pool and
    appending each successful result to the on-disk JSONL immediately."""
    missing = [s for s in slugs if s not in cache]
    if not missing:
        print("All slugs already cached — skipping embedding phase.")
        return

    print(
        f"Embedding {len(missing):,} new bird images "
        f"(concurrency={EMBED_CONCURRENCY}, cache={EMBED_CACHE_PATH.name})..."
    )
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
        futures = {
            pool.submit(_embed_one_bird, s, meta[s], data_dir): s
            for s in missing
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Embedding"
        ):
            slug, emb, err = fut.result()
            if emb is None:
                failures.append((slug, err or "unknown"))
                continue
            cache[slug] = emb
            append_embedding_cache(EMBED_CACHE_PATH, slug, emb)

    if failures:
        print(
            f"\nEmbed failures: {len(failures)} / {len(missing)} "
            "(these birds will be skipped at upsert time)",
            file=sys.stderr,
        )
        for slug, err in failures[:5]:
            print(f"  {slug}: {err}", file=sys.stderr)


def build_document(
    slug: str,
    entry: dict[str, Any],
    data_dir: pathlib.Path,
    cache: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Load text for one bird + pull its cached image embedding into a doc.

    Assumes the cache has been populated (see ``fill_embedding_cache``).
    Returns None if the embedding is missing (e.g. embed failed earlier)
    so it can be skipped cleanly rather than crashing the upsert loop.
    """
    if slug not in cache:
        print(f"  skip {slug}: no cached embedding", file=sys.stderr)
        return None

    text_path = data_dir / "text" / entry["text_file"]
    text = text_path.read_text(encoding="utf-8")
    intro, body = split_intro_body(text)

    return {
        "_id": slug,
        "bird_name": slug.replace("_", " "),
        "intro": intro,
        "body": body,
        "image_embedding": cache[slug],
    }


# -----------------------------------------------------------------------------
# Index management
# -----------------------------------------------------------------------------

# The schema here defines how the index in Pinecone parses and stores our data
# We upsert the bird names, the intro paragraphs, the body, and the image embeddings
# Note that images themselves stay on disk


def build_schema(embed_dim: int):
    return (
        SchemaBuilder()
        .add_string_field("bird_name", full_text_search={"language": "en"})
        .add_string_field("intro", full_text_search={"language": "en"})
        # Stemming on the long-prose field so "woodpeckers"/"woodpecker",
        # "pecking"/"pecks" collapse to the same lexeme for FTS matches.
        .add_string_field("body", full_text_search={"language": "en", "stemming": True})
        .add_dense_vector_field(
            "image_embedding",
            dimension=embed_dim,
            metric="cosine",
        )
        .build()
    )


def ensure_index(embed_dim: int, recreate: bool) -> None:
    if pc.preview.indexes.exists(INDEX):
        if recreate:
            print(f"Deleting existing index '{INDEX}'...")
            pc.preview.indexes.delete(INDEX)
            # Wait until it's actually gone before recreating.
            while pc.preview.indexes.exists(INDEX):
                time.sleep(2)
        else:
            print(f"Index '{INDEX}' already exists. (Pass --recreate to drop.)")
            return

    schema = build_schema(embed_dim)
    pc.preview.indexes.create(name=INDEX, schema=schema)
    print(f"Created index '{INDEX}'.")


def wait_until_ready(timeout_s: int = 300) -> None:
    print(f"Waiting for index '{INDEX}' to become ready...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = pc.preview.indexes.describe(INDEX)
        if info.status.ready:
            print(f"Index is ready (state={info.status.state!r}).")
            return
        print(f"  {info.status.state} — sleeping 5 s...")
        time.sleep(5)
    raise TimeoutError(f"Index {INDEX} not ready after {timeout_s}s")


def wait_until_searchable(idx, probe_query: str = "bird", timeout_s: int = 300) -> None:
    """Poll search() until at least one match comes back — FTS indexing is
    asynchronous, batch_upsert returning does not imply searchable.

    The default probe is corpus-specific: ``"bird"`` is in every article in this
    dataset. Pass a different sentinel for any other corpus.
    """
    print("Waiting for documents to be indexed...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = idx.documents.search(
            namespace=NAMESPACE,
            top_k=1,
            score_by=[{"type": "text", "field": "body", "query": probe_query}],
            include_fields=[],  # ids-only — lightest poll payload
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
# CLI utility for creating the index and ingesting documents. Not needed for one-off situations, but handy here!

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the bird-search-fts index and/or ingest documents into it. "
            "Separate --create-only and ingest paths so you can create first, "
            "wait for Ready, then ingest."
        ),
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help=(
            "Create the index (if absent) and wait for it to become Ready, "
            "then exit. Does not embed or upsert any documents."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of birds to ingest (default 50). Pass 0 for the full corpus.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Drop and recreate the index before ingesting. Combine with "
            "--create-only to recreate without ingesting."
        ),
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help=(
            "Delete the local embeddings-cache.jsonl before ingesting. Use "
            "after changing the embedding model or dimension."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to parsed_birds directory. "
            f"Defaults to $BIRD_DATA_DIR or '{DEFAULT_DATA_DIR}'."
        ),
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
    print(f"Using data dir: {data_dir}")

    meta = load_metadata(data_dir)

    if args.clear_cache and EMBED_CACHE_PATH.exists():
        print(f"Clearing embedding cache at {EMBED_CACHE_PATH}")
        EMBED_CACHE_PATH.unlink()

    # ---- Create-index mode (no ingestion) -----------------------------------
    if args.create_only:
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=args.recreate)
        wait_until_ready()
        print(
            f"Index '{INDEX}' is Ready. Re-run without --create-only to embed "
            "and upsert documents."
        )
        return

    # ---- Ingestion mode -----------------------------------------------------
    # If --recreate, we do the same create+wait pass first; otherwise we require
    # the index to already exist (fail-fast with a clear message).
    if args.recreate:
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=True)
        wait_until_ready()
    elif not pc.preview.indexes.exists(INDEX):
        # Auto-create on first run so `make run` / launch.py works in one shot
        # (mirrors build_text_index.py). Pass --create-only if you'd rather
        # create and wait separately before embedding.
        ensure_index(embed_dim=GEMINI_EMBED_DIMENSIONS, recreate=False)
        wait_until_ready()

    usable_slugs = filter_usable_slugs(meta, data_dir)
    if args.sample and args.sample > 0:
        slugs = usable_slugs[: args.sample]
    else:
        slugs = usable_slugs
    print(f"Preparing to ingest {len(slugs)} / {len(usable_slugs)} usable birds.")

    # ---- Embedding phase: fill cache concurrently with expo backoff ---------
    cache = load_embedding_cache(EMBED_CACHE_PATH)
    fill_embedding_cache(slugs, meta, data_dir, cache)

    # ---- Build docs (text load is cheap; embeddings come from cache) --------
    idx = pc.preview.index(name=INDEX)

    docs: list[dict[str, Any]] = []
    skipped = 0
    for slug in tqdm(slugs, desc="Building docs"):
        doc = build_document(slug, meta[slug], data_dir, cache)
        if doc is None:
            skipped += 1
            continue
        docs.append(doc)

    print(f"Built {len(docs)} documents ({skipped} skipped).")
    if not docs:
        print("No documents to upsert — exiting.")
        return

    # Batch upsert. We upload 50 articles at a time with a few concurrent workers to speed up the ingest.
    result = idx.documents.batch_upsert(
        namespace=NAMESPACE,
        documents=docs,
        batch_size=50,
        max_workers=4,
        show_progress=True,
    )
    print(
        f"\nUploaded {result.successful_item_count:,} / {result.total_item_count:,} documents"
    )
    if getattr(result, "has_errors", False):
        print(
            f"\nFailed batches: {result.failed_batch_count} / {result.total_batch_count}",
            file=sys.stderr,
        )
        # Surface the first few error messages so we actually know what's wrong.
        for err in list(getattr(result, "errors", []))[:3]:
            msg = getattr(err, "error_message", None) or str(getattr(err, "error", err))
            sample_id = err.items[0].get("_id") if getattr(err, "items", None) else "?"
            print(
                f"  batch #{getattr(err, 'batch_index', '?')} "
                f"({len(err.items)} items, first _id={sample_id!r}): {msg}",
                file=sys.stderr,
            )
        if result.successful_item_count == 0:
            raise SystemExit(
                "All upserts failed — aborting before the search-poll step."
            )

    # Poll until at least one search result comes back.
    wait_until_searchable(idx)


if __name__ == "__main__":
    main()
