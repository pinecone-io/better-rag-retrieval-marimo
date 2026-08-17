"""Bird Search — a Pinecone teaching notebook.

Open with ``marimo edit notebook.py`` to develop (code visible), or
``marimo run notebook.py`` to present (code hidden). The notebook is built to
be read top to bottom: it improves RAG *retrieval* step by step over a corpus
of North American birds, showing the exact Pinecone call behind every
technique and grading each one against a small hand-built eval set.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Bird Search — Pinecone Teaching Demo",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Bird Search: Applying retrieval techniques to improve RAG with Pinecone


    In this notebook, we'll build a sample retrieval system for a Bird Search application, and along the way, learn about how to use different retrieval mechanisms for different purposes.

    We'll build a simple evaluation set, learn a few retrieval techniques and how they perform on this set, and finally combine them with an agentic router at the end.

    We'll end with a simple interactive search application that allows you to learn each search technique side by side, passed through to a RAG chatbot, which will better contextualize the impact of the search result quality. Let's go!
    """)
    return


@app.cell
def _():
    # Standard library + our project modules. We load .env so the Pinecone and
    # Google (Gemini) API keys are visible to the SDKs.
    import pathlib

    from dotenv import load_dotenv

    load_dotenv()

    # `pc` is the shared Pinecone client; INDEX / TEXT_INDEX / NAMESPACE name
    # the two indexes we built ahead of time (see build_index.py /
    # build_text_index.py).
    from query import INDEX, TEXT_INDEX, NAMESPACE, pc

    return INDEX, NAMESPACE, TEXT_INDEX, pathlib, pc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we'll import some helpers from the repo. These are mainly to help visualize our data.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pathlib):
    # Display helpers used throughout the notebook. These are just plumbing for
    # showing birds nicely — the interesting (Pinecone) code stays visible in
    # the technique sections below.
    from urllib.parse import unquote

    IMG_DIR = pathlib.Path("parsed_birds/images")
    TEXT_DIR = pathlib.Path("parsed_birds/text")

    def pretty_name(slug):
        """'Northern_cardinal' -> 'Northern cardinal' (slugs may be URL-escaped)."""
        return unquote(slug).replace("_", " ")

    def thumb(slug, width=200):
        """A marimo image for a bird's primary photo, or a placeholder."""
        unescaped = unquote(slug)
        for path in (
            IMG_DIR / unescaped / f"{unescaped}_1.jpg",
            IMG_DIR / slug / f"{slug}_1.jpg",
        ):
            if path.exists():
                return mo.image(str(path), width=width, rounded=True)
        return mo.md(f"_(no image for `{slug}`)_")

    def read_article(slug):
        """Return a bird article's paragraphs as a list of strings."""
        for path in (TEXT_DIR / f"{unquote(slug)}.txt", TEXT_DIR / f"{slug}.txt"):
            if path.exists():
                return [p.strip() for p in path.read_text().split("\n\n") if p.strip()]
        return []

    return pretty_name, read_article, thumb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Speaking of, what is our dataset?

    We've collected a few thousand Wikipedia entries on North American birds, and their respective photos. We'll use this data to build a search application to help people who love birds, find birds and learn about them!

    Here's a sample of the data contained in the set. In addition to this, we have the entire body of the article, as well as the title of the bird itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo, pretty_name, read_article, thumb):
    # One example record: the Northern Cardinal — photo + the first couple of
    # article paragraphs. (We skip the tiny Wikipedia infobox line at the top.)
    _slug = "Northern_cardinal"
    _paras = [p for p in read_article(_slug) if len(p.split()) > 12][:2]
    _article = "\n\n".join(_paras)

    mo.hstack(
        [
            mo.vstack(
                [thumb(_slug, width=240), mo.md(f"**{pretty_name(_slug)}**")],
                align="center",
            ),
            mo.md(
                f"<div style='font-size:0.9em;line-height:1.5;'>{_article}</div>"
            ),
        ],
        align="start",
        gap=1.5,
        widths=[1, 2],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Storing the data in Pinecone: Two indexes

    We'll, of course, need a way to put this in Pinecone. We've prepared this repo to generate two indexes in Pinecone:

    - bird-search-fts, which contains the full text of the articles embedded alongside a Gemini Embedding 2 model of the images. More on this in a moment
    - bird-search-text-dense, an 800-word chunked version of the body of the corresponding articles


    Why two indexes? Pinecone's full-text-search preview allows **one dense vector field per index** — and we've spent that slot on the image embedding in `bird-search-fts`. So the chunked text embeddings need a home of their own: `bird-search-text-dense`. (It also keeps the giant pile of chunk vectors separate from the article full text, which is tidy anyway.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A["🐦 ~2,000 bird articles<br/>(Wikipedia text + a photo each)"] --> B
            A --> C
            subgraph B["Index 1 · bird-search-fts"]
                direction TB
                B1["Full-text fields:<br/>bird_name · intro · body"]
                B2["1 dense vector per bird:<br/>image_embedding (768-dim)"]
            end
            subgraph C["Index 2 · bird-search-text-dense"]
                direction TB
                C1["~3,500 body chunks<br/>(~800 words each)"]
                C2["1 dense vector per chunk:<br/>text_embedding (768-dim)"]
            end
            B --> D["Keyword search<br/>Visual search<br/>Hybrid filter + rerank"]
            C --> E["Dense semantic search<br/>Coarse-to-fine refine"]
        """
    )
    return


@app.cell(hide_code=True)
def _(INDEX, TEXT_INDEX, mo, pc):
    # Probe both indexes live, so the audience sees real status — not promises.
    # `pc.preview.indexes.describe(name)` is a real Pinecone API call.
    def _status(name):
        try:
            info = pc.preview.indexes.describe(name)
            return "✅ ready" if info.status.ready else "⏳ initializing"
        except Exception as e:  # index missing / not built yet
            return f"⚠️ {e}"

    mo.md(
        "| Index | Status | What's inside | Powers |\n"
        "|---|---|---|---|\n"
        f"| `{INDEX}` | {_status(INDEX)} | Full text + 1 image embedding / bird | Keyword · Visual · Hybrid |\n"
        f"| `{TEXT_INDEX}` | {_status(TEXT_INDEX)} | ~800-word chunks + 1 text embedding / chunk | Dense semantic · Coarse-to-fine |\n"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Meet the queries!


    Who uses a bird app? Birders, naturalists, someone who just saw something in the yard. If we wanna make a good one, we should come up with example queries that align with what these people may reasonably ask. This will form our "evaluation set".

    Note that definitions of evaluation set vary amongst practitioners. Here, we're just coming up with a small sample of queries that are useful for us to understand retrieval methods with. A "real" evaluation set might be used to version, or benchmark searches against methods. We'll avoid doing that explicitly, but it's a natural consequence of having data like this!

    Here are six queries that could be asked, and what's special about each:

    1. _"birds that trick other species into raising their chicks"_ — different words, same idea
    2. _"what bird eats Mormon crickets"_ — exact rare term
    3. _"tall pink wading bird with long curved neck"_ — looks, not words
    4. _"what gives flamingos their pink color?"_ — known bird, buried answer
    5. _"a black bird with bright spots that swoops people in Illinois"_ — looks + hard constraints
    6. _"Bright red songbird with a crest that visits backyard feeders in the eastern United States"_ — many clues at once
    """)
    return


@app.cell(hide_code=True)
def _():
    # The eval set lives in ground_truth.json: 6 queries, each with the
    # expected birds (required_doc_ids) and expected text snippets
    # (required_spans) that a good retrieval should surface.
    from evals import load_ground_truth

    GT = load_ground_truth()
    return (GT,)


@app.cell(hide_code=True)
def _(GT, mo):
    # The questions in plain language — what a user would actually ask.
    # (Targets and scoring come later; here we just meet the queries.)
    def _natural(gt):
        return gt.get("user_query") or gt["query"]

    _lines = [
        f'{i}. **"{_natural(g)}"**  —  _{g.get("teaser", "")}_'
        for i, g in enumerate(GT.values(), 1)
    ]
    mo.md(chr(10).join(_lines))
    return


@app.cell(hide_code=True)
def _(doc_recall_at_k, mo, passes, pretty_name, recall_at_k, thumb):
    # Display helpers for the technique sections: a pretty result list (with an
    # expandable full-text dropdown per hit) and a pass/fail score badge.
    import re

    def highlight(text, patterns):
        out = text or ""
        for pat in patterns or []:
            try:
                out = re.sub(f"({pat})", r"<mark>\1</mark>", out, flags=re.IGNORECASE)
            except re.error:
                pass
        return out

    def _patterns(gt):
        return [
            s["match_pattern"] if "match_pattern" in s else re.escape(s["text"])
            for s in gt.get("required_spans", [])
        ]

    def _clean(text):
        """Flatten text into one tidy paragraph: drop '---' chunk joins, collapse space."""
        text = re.sub(r"\s*-{3,}\s*", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    def snippet_for(match, gt, max_chars=320):
        body = _clean(match.get("body") or "")
        patterns = _patterns(gt)
        idx = None
        for pat in patterns:
            try:
                hit = re.search(pat, body, re.IGNORECASE)
                if hit:
                    idx = hit.start()
                    break
            except re.error:
                continue
        if idx is None:
            chunk = body[:max_chars]
        else:
            start = max(0, idx - max_chars // 3)
            chunk = ("…" if start else "") + body[start:start + max_chars]
        return highlight(chunk + ("…" if len(body) > len(chunk) else ""), patterns)

    def _full_html(match, gt):
        """The complete retrieved text, paragraph-formatted, with the spans highlighted."""
        raw = re.sub(r"\s*-{3,}\s*", "\n\n", match.get("body") or "")  # chunk joins -> para breaks
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        pats = _patterns(gt)
        return "".join(
            f"<p style='margin:0 0 0.6em;'>{highlight(p, pats)}</p>" for p in paras
        ) or "<p>(no text)</p>"

    def show_results(matches, gt, k=5):
        rows = []
        for i, m in enumerate(matches[:k], 1):
            head = mo.md(
                f"**{i}. {pretty_name(m._id)}** · score `{m._score:.3f}`\n\n"
                "<div style='font-size:0.95em;color:#1c1917;line-height:1.6;"
                "border-left:3px solid #e7e5e4;padding-left:0.75em;'>"
                f"{snippet_for(m, gt)}</div>"
            )
            full = mo.accordion({
                "Show full text": mo.md(
                    "<div style='font-size:0.9em;color:#1c1917;line-height:1.6;"
                    "max-height:340px;overflow-y:auto;border-left:3px solid #e7e5e4;"
                    f"padding-left:0.75em;'>{_full_html(m, gt)}</div>"
                )
            })
            rows.append(
                mo.hstack(
                    [thumb(m._id, width=110), mo.vstack([head, full], gap=0.35)],
                    align="start", gap=1.0, widths=[1, 4],
                )
            )
        return mo.vstack(rows or [mo.md("_(no matches)_")], gap=0.9)

    def score_badge(gt, matches, k=5):
        d, s, p = doc_recall_at_k(gt, matches, k), recall_at_k(gt, matches, k), passes(gt, matches, k)
        return mo.md(
            f"**Score** — {'✅ pass' if p else '❌ fail'} · "
            f"doc recall@{k} `{d:.2f}` · span recall@{k} `{s:.2f}`"
        )

    def query_string(gt):
        q = gt["query"]
        return q if isinstance(q, str) else (q.get("semantic") or q.get("visual") or "")

    return query_string, score_badge, show_results


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How we'll score each result

    Next, we'll need to label the documents and their parts that map to each query. We'll take a simple approach, and look for the following metrics on a document level, and chunk level.

    - **doc recall** — did the expected bird(s) show up?
    - **span recall** — did the answer text show up?
    - **pass** — surfaced enough to call it a win
    """)
    return


@app.cell(hide_code=True)
def _(GT, mo, pretty_name):
    # The answer key: for each question, the bird(s) and text a good search
    # should surface. This is exactly what the scores are measured against.
    def _natural(gt):
        return gt.get("user_query") or gt["query"]

    def _birds_readable(gt):
        birds = [pretty_name(b) for b in gt.get("required_doc_ids", [])]
        joined = " / ".join(birds) if birds else "—"
        if gt.get("minimum_doc_recall", len(birds)) < len(birds):
            joined += f" _(any {gt['minimum_doc_recall']})_"
        return joined

    def _text_readable(gt):
        out = []
        for s in gt.get("required_spans", []):
            if "text" in s:
                out.append(f'"{s["text"]}"')
            elif "match_pattern" in s:
                out.append(" / ".join(s["match_pattern"].split("|")[:6]))
        return "  •  ".join(out) or "—"

    _rows = [
        "**The answer key** — the bird(s) and text each question should surface:",
        "",
        "| # | Question | Expected bird(s) | Answer text we look for |",
        "|---|---|---|---|",
    ]
    for _i, _g in enumerate(GT.values(), 1):
        _rows.append(f"| {_i} | {_natural(_g)} | {_birds_readable(_g)} | {_text_readable(_g)} |")
    mo.md(chr(10).join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part A · Three primitives

    We first start with three retrieval methods that form the basis of our strategy: dense search, full text search, and image search.

    Later, we'll combine these to form more complicated searches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Dense semantic retrieval — match by *meaning* over chunks

    Dense retrieval is great for when addressing user queries that run into vocabulary mismatch problems, especially in technical domains. Here, our wikipedia articles are pretty technical, and we have users that are asking questions over that data.

    The chances that a passage in a wikipedia article contains the same words in the same order are quite low, so being able to abstract the meaning out of the query is quite helpful.

    Additionally, the user is asking for a specific quality about this bird, which implies finding the section relevant for the question. Our chunked, dense index is great for this!


    **User asks —** _birds that trick other species into raising their chicks_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's our dense search function, which uses our helpers to embed our query, then hit Pinecone. Pretty straightforward!
    """)
    return


@app.cell(hide_code=True)
def _():
    # Shared plumbing for the technique sections. `embed_text` turns text into a
    # Gemini vector; `BirdMatch` is a tiny match object the eval/render code
    # understands; `chunks_to_birds` rolls chunk-level hits up to one per bird.
    # (We alias query._BirdMatch -> BirdMatch: marimo treats leading-underscore
    # top-level names as cell-private, so a shared import must not start with "_".)
    from query import embed_text
    from query import _BirdMatch as BirdMatch
    from evals import recall_at_k, doc_recall_at_k, passes

    def chunks_to_birds(chunk_matches, top_k):
        """Dense-text search returns article *chunks*; group them by bird,
        sum the chunk scores, and keep the best chunks' text as the body."""
        def sc(m):
            s = getattr(m, "_score", None)
            return s if s is not None else (getattr(m, "score", 0.0) or 0.0)

        by_bird = {}
        for m in chunk_matches:
            slug = m.get("slug") or m._id.split("#")[0]
            by_bird.setdefault(slug, []).append(m)

        birds = []
        for slug, chunks in by_bird.items():
            chunks.sort(key=sc, reverse=True)
            birds.append(BirdMatch(
                _id=slug,
                score=sum(sc(c) for c in chunks),
                fields={
                    "bird_name": chunks[0].get("bird_name") or slug.replace("_", " "),
                    "body": "\n\n".join((c.get("chunk_text") or "") for c in chunks[:3]),
                },
            ))
        return sorted(birds, key=lambda b: b._score, reverse=True)[:top_k]

    return (
        BirdMatch,
        chunks_to_birds,
        doc_recall_at_k,
        embed_text,
        passes,
        recall_at_k,
    )


@app.cell
def _(NAMESPACE, TEXT_INDEX, chunks_to_birds, embed_text, pc):
    def dense_semantic_search(query, top_k=5):
        # 1. Embed the text query into Gemini's 768-dim space.
        query_vector = embed_text(query)

        # 2. Ask Pinecone for the article chunks whose text_embedding is closest.
        text_index = pc.preview.index(name=TEXT_INDEX)
        response = text_index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k * 5,          # over-fetch chunks; we roll them up to birds
            score_by=[{
                "type": "dense_vector",
                "field": "text_embedding",
                "values": query_vector,
            }],
            include_fields=["slug", "bird_name", "chunk_text"],
        )

        # 3. Several chunks can come from the same bird — combine to one per bird.
        return chunks_to_birds(response.matches, top_k)

    return (dense_semantic_search,)


@app.cell
def _(GT, dense_semantic_search, mo, score_badge, show_results):
    #
    q1_gt = GT["q1_brood_parasites"]
    q1_matches = dense_semantic_search(q1_gt["query"], top_k=5)

    mo.vstack([show_results(q1_matches, q1_gt), score_badge(q1_gt, q1_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Keyword (BM25) — match the *exact words*

    For our search application, we probably also want to support precise searches that let us pluck relevant articles quickly. This means we need some way to support "Full text" search, which relies heavily on keywords.

    This next query about Mormon crickets is chosen as there's exactly 1 article in the whole database that refers to this, the one about the California Gull!

    **User asks —** _what bird eats Mormon crickets?_

    "Mormon crickets" is a rare exact phrase in just one article. Keyword pinpoints it — the California gull, famous for devouring a plague of them. Dense drifts to generic insect-eaters and **misses the gull entirely**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's our keyword search codeblock, which is pretty clean. We search over the body of the articles and return the results.
    """)
    return


@app.cell
def _(INDEX, NAMESPACE, pc):
    def keyword_search(query, field="body", top_k=5):
        # Classic full-text search (BM25) on the MAIN index — matches the literal
        # words in a text field. Best for names, rare phrases, and exact terms.
        index = pc.preview.index(name=INDEX)
        response = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{"type": "text", "field": field, "query": query}],
            include_fields=["bird_name", "intro", "body"],
        )
        return list(response.matches)

    return (keyword_search,)


@app.cell
def _(GT, keyword_search, mo, score_badge, show_results):
    q2_gt = GT["q2_mormon_crickets"]
    q2_matches = keyword_search(q2_gt["query"], field="body", top_k=5)

    mo.vstack([show_results(q2_matches, q2_gt), score_badge(q2_gt, q2_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Visual cross-modal — match by *looks*

    Next, most bird-spotting occurs when someone is outside looking at birds! So what happens if you see one, and want to describe it?

    Image embeddings work great here, especially ones that allow us to convert text to images and vice versa.

    The Gemini Embedding 2 model specifically is great for this, and we'll use that property to enable answering our third query:

    **User asks —** _tall pink wading bird with long curved neck_

    Articles rarely say "pink". Embed the TEXT query and score it against stored bird PHOTOS — same vector space, so words can match images. Keyword has no "pink" to find.
    """)
    return


@app.cell
def _(INDEX, NAMESPACE, embed_text, pc):
    def visual_search(query, top_k=5):
        # Cross-modal: embed the TEXT query with Gemini, then score it against the
        # stored IMAGE embeddings. Text and images share one Gemini vector space,
        # so "pink wading bird" can match a flamingo photo directly.
        query_vector = embed_text(query)
        # note that we use the same embedding model!

        index = pc.preview.index(name=INDEX)
        response = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            score_by=[{"type": "dense_vector", "field": "image_embedding", "values": query_vector}],
            include_fields=["bird_name", "intro", "body"],
        )
        return list(response.matches)

    return (visual_search,)


@app.cell
def _(GT, mo, score_badge, show_results, visual_search):
    q3_gt = GT["q3_pink_wading_bird"]
    q3_matches = visual_search(q3_gt["query"], top_k=5)

    mo.vstack([show_results(q3_matches, q3_gt), score_badge(q3_gt, q3_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part B · Combining the primitives

    Real questions from real birders rarely fit neatly into one box. Someone might describe what a bird *looks* like AND tell you where they saw it, or name a bird and then ask something specific about it.

    The three primitives from Part A — dense, keyword, and image search — are our building blocks. Now we'll start combining them to handle these messier, more realistic queries: a hard filter paired with a visual rerank (**hybrid**), narrowing to a bird and then digging inside its article (**coarse-to-fine**), and finally letting Claude pick and combine the tools for us (**agentic**).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hybrid — filter, then rerank by *looks*

    Sometimes a query has both a hard requirement and a soft one. Our birder saw a black bird with bright spots that swooped at them in Illinois — "Illinois" and "swoop" are facts that *must* match, but "black bird with bright spots" is really about appearance.

    So we combine two primitives in one call. First a full text **filter** keeps only the articles that mention both "swoop" and "illinois" ($match_all — both have to appear), then we **visually rerank** whatever survives against the photos.

    **User asks —** _a black bird with bright spots that swoops people in Illinois_

    This is the only technique that cracks this one: image search alone confuses all the black birds, and keyword search alone ignores what it looks like. Together they pin down the red-winged blackbird.
    """)
    return


@app.cell
def _(INDEX, NAMESPACE, embed_text, pc):
    def hybrid_filter_visual(filter_terms, visual_query, top_k=5):
        # Hard requirement + soft ranking in ONE call: every term in filter_terms
        # MUST appear in the body ($match_all), and the survivors are ranked by
        # visual similarity to the description.
        query_vector = embed_text(visual_query)
        index = pc.preview.index(name=INDEX)
        response = index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k,
            filter={"body": {"$match_all": " ".join(filter_terms)}},
            score_by=[{"type": "dense_vector", "field": "image_embedding", "values": query_vector}],
            include_fields=["bird_name", "intro", "body"],
        )
        return list(response.matches)

    return (hybrid_filter_visual,)


@app.cell
def _(GT, hybrid_filter_visual, mo, score_badge, show_results):
    q5_gt = GT["q5_redwinged_blackbird_hybrid"]
    q5_query = q5_gt["query"]
    q5_matches = hybrid_filter_visual(q5_query["filter_terms"], q5_query["visual"], top_k=5)

    mo.vstack([show_results(q5_matches, q5_gt), score_badge(q5_gt, q5_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Coarse-to-fine — scope to a bird, then refine

    Sometimes the user already knows *which* bird they care about, and they're really asking a question about its article. Here they've named the flamingo and want a specific fact buried somewhere inside the page.

    So we go in two stages. First, **coarse**: narrow down to the right bird with a keyword gate on "flamingo". Then, **fine**: run a dense search *inside* those candidates' chunks to surface the exact passage that answers the question.

    **User asks —** _what gives flamingos their pink color?_

    A fair warning: on a corpus this small, plain dense search already lands on the answer — so you won't see coarse-to-fine "win" on the scoreboard. The point is the **shape** of it (filter → refine). At scale, when lots of articles mention a "pink color", gating to the right bird first is what keeps you from confidently pulling a great-sounding passage out of the *wrong* one.
    """)
    return


@app.cell
def _(INDEX, NAMESPACE, TEXT_INDEX, chunks_to_birds, embed_text, pc):
    def search_within(filter_terms, semantic_query, top_k=5, candidate_pool=20):
        # STAGE 1 (coarse): keyword-gate the MAIN index down to candidate birds.
        main = pc.preview.index(name=INDEX)
        gate = main.documents.search(
            namespace=NAMESPACE,
            top_k=candidate_pool,
            score_by=[{"type": "text", "field": "body", "query": " ".join(filter_terms)}],
            include_fields=["bird_name"],
        )
        candidate_slugs = [m._id for m in gate.matches]

        # STAGE 2 (fine): dense-semantic search the article CHUNKS, restricted to
        # those candidates, to surface the passage that answers the question.
        query_vector = embed_text(semantic_query)
        text_index = pc.preview.index(name=TEXT_INDEX)
        refine = text_index.documents.search(
            namespace=NAMESPACE,
            top_k=top_k * 5,
            filter={"slug": {"$in": candidate_slugs}},
            score_by=[{"type": "dense_vector", "field": "text_embedding", "values": query_vector}],
            include_fields=["slug", "bird_name", "chunk_text"],
        )
        return chunks_to_birds(refine.matches, top_k)

    return (search_within,)


@app.cell
def _(GT, mo, score_badge, search_within, show_results):
    q4_gt = GT["q4_flamingo_pink_color"]
    q4_query = q4_gt["query"]
    q4_matches = search_within(q4_query["filter_terms"], q4_query["semantic"], top_k=5)

    mo.vstack([show_results(q4_matches, q4_gt), score_badge(q4_gt, q4_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Agentic — let Claude route and merge

    Up to now *we've* been the ones picking which technique fits each query. But a real search box doesn't know in advance what kind of question it's going to get. Our last query has a bit of everything: an appearance ("bright red, with a crest"), a behavior ("visits backyard feeders"), and a region ("eastern US") — and any single technique only chases one of those facets.

    So we hand the wheel to Claude. It reads the messy question, breaks it into facets, **picks a tool for each one and fills in the arguments**, runs them all, and merges the results — the same kind of routing we just did by hand, but automatic.

    **User asks —** _bright red songbird with a crest that visits backyard feeders in the eastern United States_

    It won't out-retrieve a well-chosen single tool — what it removes is the *choosing*. Watch the plan it writes below: that's Claude deciding how to search.
    """)
    return


@app.cell
def _():
    # The agent's brain: a catalog of the FIVE primitives we built above. Claude
    # reads the question, splits it into facets, and picks a tool (with args) for
    # each — one call in, a JSON plan out. No tool-calling loop.
    import anthropic
    import json

    AGENT_MODEL = "claude-sonnet-5"

    AGENT_TOOLS = """\
    You route bird-search questions across these five retrieval tools. Break the
    question into 1-4 facets and pick the best tool for each. Return JSON only.

    Tools (and the args each takes):
    - dense_semantic_search(query)   -> match by MEANING: paraphrases, concepts.
    - keyword_search(query)          -> match EXACT words: names, rare terms.
    - visual_search(query)           -> match by APPEARANCE: color, shape, size.
    - hybrid_filter_visual(filter_terms, visual_query)
          -> REQUIRE words in the text, then rank survivors by looks.
             filter_terms is a LIST of a few must-appear words, e.g. ["illinois", "swoop"].
    - search_within(filter_terms, semantic_query)
          -> scope to a bird via filter_terms (a LIST of words), then dense-refine inside it.

    Guidance: use visual_search for appearance (color, shape); keyword_search or
    dense_semantic_search for behavior, diet, or region; hybrid_filter_visual only
    when a hard text constraint AND appearance both matter, with filter_terms of
    1-2 distinctive words.

    Output JSON, nothing else:
    {"steps": [{"technique": "<tool>", "args": {...}, "why": "<short phrase>"}]}
    """

    def decompose(question):
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=4096,           # shared with thinking on newer models
            system=AGENT_TOOLS,
            messages=[{"role": "user", "content": question}],
        )
        # Pull the text blocks out by type. Newer Claude models think before they
        # answer, so the first block may be a thinking block rather than the JSON.
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        if text.startswith("```"):     # strip ```json fences if present
            text = text.removeprefix("```json").removeprefix("```").rsplit("```", 1)[0].strip()
        return json.loads(text)["steps"]

    return decompose, json


@app.cell(hide_code=True)
def _(GT, decompose, json, mo):
    # One Claude call turns the question into a routing plan. Show the facets,
    # the tool Claude picked for each, and the arguments it derived.
    q6_gt = GT["q6_cardinal_eastern_feeders"]
    q6_plan = decompose(q6_gt["query"])

    _FRIENDLY = {
        "dense_semantic_search": "Dense semantic",
        "keyword_search": "Keyword (BM25)",
        "visual_search": "Visual cross-modal",
        "hybrid_filter_visual": "Hybrid filter + visual",
        "search_within": "Coarse-to-fine",
    }
    _rows = [
        "**Claude's plan** — one call, decomposed into facets and routed:",
        "",
        "| Facet (why) | Technique | Arguments Claude derived |",
        "|---|---|---|",
    ]
    for _s in q6_plan:
        _rows.append(
            f"| {_s.get('why', '')} | **{_FRIENDLY.get(_s['technique'], _s['technique'])}** | `{json.dumps(_s['args'], ensure_ascii=False)}` |"
        )
    mo.md(chr(10).join(_rows))
    return q6_gt, q6_plan


@app.cell
def _(
    BirdMatch,
    dense_semantic_search,
    hybrid_filter_visual,
    keyword_search,
    search_within,
    visual_search,
):
    # Run the plan: send each step to the matching primitive from Part A, then
    # merge. We min-max normalize each step's scores and sum them per bird, so a
    # bird that satisfies several facets outranks one that nails only a single tool.
    def run_agentic(plan, top_k=5):
        tools = {
            "dense_semantic_search": lambda a: dense_semantic_search(a["query"], top_k=20),
            "keyword_search":        lambda a: keyword_search(a["query"], top_k=20),
            "visual_search":         lambda a: visual_search(a["query"], top_k=20),
            "hybrid_filter_visual":  lambda a: hybrid_filter_visual(a["filter_terms"], a["visual_query"], top_k=20),
            "search_within":         lambda a: search_within(a["filter_terms"], a["semantic_query"], top_k=20),
        }

        # 1. Execute every step (skip any the model mis-specified).
        step_results = []
        for step in plan:
            fn = tools.get(step["technique"])
            if fn is None:
                continue
            args = dict(step.get("args", {}))
            if isinstance(args.get("filter_terms"), str):   # tolerate a string
                args["filter_terms"] = args["filter_terms"].split()
            try:
                step_results.append(fn(args))
            except Exception:
                continue

        # 2. Merge by normalized-score sum across steps.
        scores, reps, counts = {}, {}, {}
        for matches in step_results:
            raw = [getattr(m, "_score", 0.0) or 0.0 for m in matches]
            if not raw:
                continue
            lo, hi = min(raw), max(raw)
            span = (hi - lo) or 1.0
            for m, s in zip(matches, raw):
                scores[m._id] = scores.get(m._id, 0.0) + (s - lo) / span
                counts[m._id] = counts.get(m._id, 0) + 1
                # Keep the richest body for each bird. The FTS tools return the whole
                # article; the chunk tools return only their top chunks. Taking
                # whichever step ran last would let a chunk tool truncate the text
                # that the score badge and the RAG context both read.
                prev = reps.get(m._id)
                if prev is None or len(m.get("body") or "") > len(prev.get("body") or ""):
                    reps[m._id] = m

        # 3. Rank, wrap in BirdMatch so the merged score is what shows.
        ranked = sorted(scores, key=lambda i: (scores[i], counts[i]), reverse=True)
        out = []
        for _id in ranked[:top_k]:
            m = reps[_id]
            fields = {f: (m.get(f) if hasattr(m, "get") else None)
                      for f in ("bird_name", "intro", "body", "chunk_text")}
            out.append(BirdMatch(_id=_id, score=scores[_id], fields=fields))
        return out

    return (run_agentic,)


@app.cell
def _(mo, q6_gt, q6_plan, run_agentic, score_badge, show_results):
    # Same query, now answered by the agent: it ran the plan above and merged the
    # results from each tool into one ranking.
    q6_matches = run_agentic(q6_plan, top_k=5)

    mo.vstack([show_results(q6_matches, q6_gt), score_badge(q6_gt, q6_matches)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Recap · every technique on every question

    We've now watched each technique handle the one query it was built for. The natural next question: what happens when we run *every* technique against *every* query? This scorecard is that bird's-eye view — where each method shines, and where it falls flat.

    **How to read it.** Each **row** is one of our six queries, and each **column** is a retrieval technique. The value in a cell is the **rank of the bird we were hoping for** — how far down that technique's results the correct bird showed up:

    - **1** — nailed it; the right bird came back *first*.
    - **2–10** — found it, but lower down the list.
    - **·** — never showed up in the top 10. A miss.
    - **—** — doesn't apply: this technique needs structured input (like filter terms) that this query doesn't give it.
    - **★** — marks the technique each query was *designed* to showcase (the ones we walked through above). The last query has no ★: its technique — the **agentic router** — isn't a column here, because it sits *on top* of these five and orchestrates them.

    Heads up: unlike the sections above, there's nothing to click into here — we're not re-showing the actual results, just summarizing each run with that one rank number. So treat it as shorthand for "how well did it do?", and scroll back up if you want to see the birds themselves.

    **What to notice.** The techniques **overlap** more than you'd expect — on the easier queries, several methods all land the right bird. The decisive, this-tool-and-only-this-tool wins are actually pretty narrow: only the hybrid filter cracks the Illinois blackbird, only visual clearly beats keyword on the pink wader, and the flamingo query's real payoff is the *passage* inside the article, not just finding the bird. There's no tidy "one tool per query" rule here. The real skill is knowing the tradeoffs — and reaching for the right tool, or combining a few, for the job in front of you.
    """)
    return


@app.cell(hide_code=True)
def _(
    GT,
    dense_semantic_search,
    hybrid_filter_visual,
    keyword_search,
    mo,
    query_string,
    search_within,
    visual_search,
):
    # Honest scorecard: where does the expected bird land in each technique's ranking?
    # Rows = the six queries (readable text), columns = techniques. The cell value is
    # the RANK of the expected bird (1 = top hit). String techniques take any query;
    # structured ones (coarse-to-fine, hybrid) only run where the query supplies
    # filter terms, else "—". A ★ marks the technique each query was built to showcase.
    def _q(gt): return query_string(gt)
    def _dense_fn(gt):   return dense_semantic_search(_q(gt), top_k=10)
    def _keyword_fn(gt): return keyword_search(_q(gt), field="body", top_k=10)
    def _visual_fn(gt):  return visual_search(_q(gt), top_k=10)
    def _within_fn(gt):
        q = gt["query"]
        if not (isinstance(q, dict) and q.get("filter_terms")): return None
        return search_within(q["filter_terms"], q.get("semantic") or q.get("visual"), top_k=10)
    def _hybrid_fn(gt):
        q = gt["query"]
        if not (isinstance(q, dict) and q.get("filter_terms")): return None
        return hybrid_filter_visual(q["filter_terms"], q.get("visual") or q.get("semantic"), top_k=10)

    def _rank(gt, matches):
        req = set(gt.get("required_doc_ids", []))
        for i, m in enumerate(matches or [], 1):
            if m._id in req:
                return i
        return None

    def _readable(gt):
        if gt.get("user_query"):
            return gt["user_query"]
        q = gt["query"]
        return q if isinstance(q, str) else str(q)

    # (column header, function, the intended_technique key it corresponds to)
    _TECHNIQUES = [
        ("Dense", _dense_fn, "text_semantic"),
        ("Keyword", _keyword_fn, "text_fts_body"),
        ("Visual", _visual_fn, "visual"),
        ("Coarse→fine", _within_fn, "search_within"),
        ("Hybrid", _hybrid_fn, "filter_visual"),
    ]
    _qids = list(GT.keys())
    _header = "| Query | " + " | ".join(h for h, _f, _k in _TECHNIQUES) + " |"
    _lines = [_header, "|" + "---|" * (len(_TECHNIQUES) + 1)]
    for _qid in _qids:
        _gt = GT[_qid]
        _intended = _gt.get("intended_technique")
        _cells = []
        for _hdr, _fn, _key in _TECHNIQUES:
            _m = _fn(_gt)
            if _m is None:
                _val = "—"
            else:
                _r = _rank(_gt, _m)
                _val = "·" if _r is None else (f"**{_r}**" if _r == 1 else str(_r))
            if _key == _intended:
                _val = f"{_val} ★"
            _cells.append(_val)
        _lines.append(f"| _{_readable(_gt)}_ | " + " | ".join(_cells) + " |")
    mo.md(chr(10).join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Try it yourself

    Type any query, pick a technique, and see what comes back. Flip on **Generate a RAG answer** to watch retrieval feed an LLM — good retrieval in, good answer out.
    """)
    return


@app.cell(hide_code=True)
def _():
    from generation import generate

    return (generate,)


@app.cell
def _(mo):
    pb_query = mo.ui.text(value="a small iridescent green hummingbird", label="Query", full_width=True)
    pb_tech = mo.ui.dropdown(
        ["Dense semantic", "Keyword (BM25)", "Visual cross-modal", "Agentic (route + merge)"],
        value="Dense semantic", label="Technique",
    )
    pb_k = mo.ui.slider(3, 10, value=5, label="top_k")
    pb_rag = mo.ui.switch(label="Generate a RAG answer")
    mo.vstack([pb_query, mo.hstack([pb_tech, pb_k, pb_rag], justify="start", gap=1.5)])
    return pb_k, pb_query, pb_rag, pb_tech


@app.cell
def _(
    decompose,
    dense_semantic_search,
    generate,
    keyword_search,
    mo,
    pb_k,
    pb_query,
    pb_rag,
    pb_tech,
    run_agentic,
    show_results,
    visual_search,
):
    import json as _json_pb

    _q = (pb_query.value or "").strip()

    # Friendly labels for the five inline tools the agent routes to.
    _FRIENDLY_PB = {
        "dense_semantic_search": "Dense semantic",
        "keyword_search": "Keyword (BM25)",
        "visual_search": "Visual cross-modal",
        "hybrid_filter_visual": "Hybrid filter + visual",
        "search_within": "Coarse-to-fine",
    }

    if not _q:
        _out = mo.md("_Type a query above to search._")
    else:
        _k = pb_k.value
        _plan_block = None
        if pb_tech.value == "Dense semantic":
            _m = dense_semantic_search(_q, top_k=_k)
        elif pb_tech.value == "Keyword (BM25)":
            _m = keyword_search(_q, field="body", top_k=_k)
        elif pb_tech.value == "Visual cross-modal":
            _m = visual_search(_q, top_k=_k)
        else:
            # Agentic: the same inline agent from the section above — decompose, then run.
            _plan = decompose(_q)
            _m = run_agentic(_plan, top_k=_k)
            _plan_lines = [
                "**Claude’s plan** — one call, decomposed into facets and routed:",
                "",
                "| Facet (why) | Technique | Arguments Claude derived |",
                "|---|---|---|",
            ]
            for _step in _plan:
                _label = _FRIENDLY_PB.get(_step["technique"], _step["technique"])
                _args = _json_pb.dumps(_step["args"], ensure_ascii=False)
                _plan_lines.append(f"| {_step.get('why', '')} | **{_label}** | `{_args}` |")
            _plan_block = mo.callout(mo.md(chr(10).join(_plan_lines)), kind="neutral")

        _blocks = [show_results(_m, {}, k=_k)]
        if _plan_block is not None:
            _blocks.insert(0, _plan_block)
        if pb_rag.value and _m:
            _ans = generate(_q, _m).answer
            _blocks.insert(0, mo.callout(mo.md(f"**Answer:** {_ans}"), kind="info"))
        _out = mo.vstack(_blocks)
    _out
    return


if __name__ == "__main__":
    app.run()
