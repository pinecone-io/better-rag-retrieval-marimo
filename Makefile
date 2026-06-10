.DEFAULT_GOAL := help
.PHONY: help run sync demo present test clean

help:        ## Show available commands.
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run:         ## One command: sync deps, build any missing indexes, open notebook.
	uv sync
	uv run python launch.py

sync:        ## Install or update Python dependencies into .venv.
	uv sync

demo:        ## Open the notebook in edit mode (code visible).
	uv run marimo edit notebook.py

present:     ## Present the notebook (code hidden — for the talk).
	uv run marimo run notebook.py

test:        ## Run all pytest smoke tests.
	uv run python -m pytest tests/ -v

clean:       ## Remove the venv and on-disk embedding caches.
	rm -rf .venv embeddings-cache.jsonl text-embeddings-cache.jsonl
