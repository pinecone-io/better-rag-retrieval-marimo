# Bird Search — a Pinecone teaching demo 🐦

A lecture demo on **improving RAG retrieval** with Pinecone, over a corpus of
~2,000 North American bird articles (Wikipedia text + a photo each). The
deliverable is a **Marimo notebook** that teaches *when to reach for which
retrieval technique* — keyword, dense semantic, visual cross-modal, and ways to
combine them — by showing the real Pinecone call behind each and grading it
against a small hand-built eval set.

## Quickstart

**Prerequisites:**

- **[uv](https://docs.astral.sh/uv/)** — the Python runner this repo uses (install command below).
- **Three API keys** — Pinecone, Google Gemini, and Anthropic Claude. Getting them
  takes about five minutes; see [Get your API keys](#get-your-api-keys) below.

Install uv if you don't have it, then set up and launch:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (skip if you already have it)
cp .env.example .env                              # paste your three keys into .env (see below)
make run                                          # installs deps, builds missing indexes, opens the notebook
```

`make run` handles the rest — no manual dependency install, and it builds the two
Pinecone indexes for you if they don't exist yet.

> **First run builds two Pinecone indexes.** The text-dense index is quick
> (~3–5 min). The image index embeds ~2,000 bird photos with Gemini and takes
> **15–30 min** the first time; it's cached on disk, so later runs are instant.
> You can interrupt and resume — embeddings are cached as they're computed.

## Get your API keys

You need three keys. Create each one (all have a free or low-cost tier), then paste
it into `.env` next to the matching variable.

| Service | Used for | Where to get it | `.env` variable |
|---|---|---|---|
| **Pinecone** | The vector + full-text indexes | [app.pinecone.io](https://app.pinecone.io) | `PINECONE_API_KEY` |
| **Google Gemini** | Embeddings (text **and** images) | [aistudio.google.com](https://aistudio.google.com) | `GOOGLE_API_KEY` |
| **Anthropic Claude** | The agentic router + RAG answers | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` |

Step by step:

1. **Pinecone** — create a free account at [app.pinecone.io](https://app.pinecone.io).
   In the left sidebar open **API Keys**, click **Create API key**, and copy the value.
   The free Starter plan is enough for this demo.
2. **Google Gemini** — go to [Google AI Studio](https://aistudio.google.com), click
   **Get API key** (top right) → **Create API key** (pick or create a Google Cloud
   project if asked), and copy it. Gemini's free tier covers this demo.
3. **Anthropic Claude** — sign in at [console.anthropic.com](https://console.anthropic.com),
   open **API Keys**, click **Create Key**, and copy it. Anthropic needs a small
   prepaid balance (a few dollars covers the whole demo). Claude is only used by the
   agentic section and the RAG answers — the three core retrieval techniques run on
   Pinecone + Gemini alone, so you can start without it.

Your `.env` should end up looking like this (keys shortened):

```bash
PINECONE_API_KEY=pcsk_xxxxxxxx...
GOOGLE_API_KEY=AIzaxxxxxxxx...
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx...
```

> `.env` is already in `.gitignore` — don't commit it or share it.

## Commands

```bash
make demo        # open the notebook in edit mode (code visible)
make present     # presentation mode — code hidden, for the talk
make test        # run pytest smoke tests
make clean       # remove .venv and on-disk embedding caches
make help        # list available commands
```

## What's in the notebook

It reads top to bottom. After meeting the data, the two indexes, and the eval
set, it walks **three primitives** and then **three ways to combine them** —
each shown with its live Pinecone call, results, and score:

| | Technique | Example question | The point |
|---|---|---|---|
| **Primitives** | Dense semantic | *birds that trick others into raising their chicks* | matches by **meaning** — finds the brood parasites though the words never appear |
| | Keyword (BM25) | *what bird eats Mormon crickets?* | matches the **exact rare term** — pinpoints the California gull where semantics drifts |
| | Visual cross-modal | *tall pink wading bird* | matches by **looks** — the text never says "pink", the photo does |
| **Compositions** | Hybrid filter + visual | *black bird that swoops in Illinois* | a hard text **filter** + a **visual rerank** in one call |
| | Coarse-to-fine | *what gives flamingos their pink color?* | **scope** to an entity, then **refine** to the answer passage |
| | Agentic | *red crested feeder bird in the eastern US* | Claude **routes** the facets across the tools and merges |

It ends with a **recap scorecard** (every technique × every question) and an
**interactive builder** — type a query, pick a technique, see results and an
optional RAG answer.

## Architecture

Two Pinecone indexes back the demo:

| Index | Holds | Used for |
|---|---|---|
| `bird-search-fts` | Full-text fields (`bird_name`, `intro`, `body`) + one dense `image_embedding` per bird | Keyword search, visual cross-modal, hybrid (filter + image rerank) |
| `bird-search-text-dense` | ~800-word chunks of each article body + one dense `text_embedding` per chunk | Dense semantic search, and coarse-to-fine refine |

Two indexes (not one) because Pinecone's full-text-search preview API currently
allows **at most one dense vector field per index** — we use that slot for image
embeddings on the main index, and a sister index holds the text embeddings.

## File map

```
notebook.py            # The Marimo demo — the main artifact (authors every retrieval call inline)
launch.py              # `make run`: checks env, builds missing indexes, opens notebook
query.py               # Shared plumbing the notebook imports — Pinecone client + Gemini embed helper
generation.py          # Claude RAG step — answers from retrieved articles
evals.py               # Recall metrics behind the notebook's scorecard
ground_truth.json      # The 6 eval queries with expected birds and answer spans
build_index.py         # Build the main image+FTS index
build_text_index.py    # Build the sister text-dense index (chunks each article)
parsed_birds/          # ~2,000 Wikipedia bird articles + photos
tests/                 # Pytest smoke tests
Makefile               # One-command operations
pyproject.toml         # uv-managed dependencies
.env.example           # Required API keys
```

## Dataset

`parsed_birds/` contains North American bird articles scraped from Wikipedia.
Each bird has a text file (`text/<slug>.txt`) and a primary photo
(`images/<slug>/<slug>_1.jpg`). Set `BIRD_DATA_DIR` to override the default path.

## Why Gemini Embedding 2

[Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) is
multimodal — text and images land in the same vector space, so a text query like
*"tall pink wading bird"* is directly comparable to a bird's photo embedding. No
two-stage captioning pipeline needed. We use 768 dimensions with cosine
similarity for both modalities.

## Learn more

- [Pinecone Full-Text Search docs](https://docs.pinecone.io/guides/search/full-text-search)
- [Marimo docs](https://docs.marimo.io)

## License

MIT
