"""One-command launcher for the bird-search demo.

Workflow when you run ``make run`` (or ``uv run python launch.py``):

1. Verify the three API keys are set in ``.env``.
2. Probe Pinecone for both indexes the demo needs.
3. Build any index that's missing (image+FTS first, then text-dense).
4. Launch the Marimo notebook in edit mode.

Idempotent — re-running after the indexes are built just opens the notebook.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_KEYS = ("PINECONE_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")
MAIN_INDEX = "bird-search-fts"
TEXT_INDEX = "bird-search-text-dense"


def _check_env() -> None:
    if not Path(".env").exists():
        sys.exit(
            "❌ No .env file in the current directory.\n"
            "   Copy .env.example to .env and fill in your API keys."
        )
    load_dotenv()
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"❌ Missing API key(s) in .env: {', '.join(missing)}\n"
            f"   Edit .env and rerun."
        )
    print("✓ All API keys present.")


def _list_indexes() -> set[str]:
    from pinecone import Pinecone
    pc = Pinecone()
    return {i.name for i in pc.list_indexes()}


def _build_if_missing(have: set[str]) -> None:
    if MAIN_INDEX not in have:
        print(f"\n⏳ {MAIN_INDEX!r} not found — building it now.")
        print("   This embeds ~2,000 bird photos with Gemini-2. Expect 15-30 minutes.")
        print("   You can interrupt and resume; embeddings are cached on disk.\n")
        subprocess.check_call(
            [sys.executable, "build_index.py", "--sample", "0"]
        )
    else:
        print(f"✓ {MAIN_INDEX!r} already exists.")

    if TEXT_INDEX not in have:
        print(f"\n⏳ {TEXT_INDEX!r} not found — building it now.")
        print("   This chunks each article into ~800-word passages and embeds")
        print("   each chunk. Expect 3-5 minutes.\n")
        subprocess.check_call(
            [sys.executable, "build_text_index.py", "--sample", "0"]
        )
    else:
        print(f"✓ {TEXT_INDEX!r} already exists.")


def _launch_notebook() -> None:
    print("\n🚀 Launching the notebook...\n")
    # Run marimo from the active interpreter's environment (the venv), not a
    # global install. execvp replaces this process so marimo runs in foreground.
    os.execvp(sys.executable, [sys.executable, "-m", "marimo", "edit", "notebook.py"])


def main() -> None:
    _check_env()
    have = _list_indexes()
    _build_if_missing(have)
    _launch_notebook()


if __name__ == "__main__":
    main()
