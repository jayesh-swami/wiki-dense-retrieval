# wiki-hybrid-retrieval Makefile
# Run `make` to see all targets. Benchmarks have their own Makefile: `make -C benchmark`.
# Recipes must be indented with tabs, not spaces.

WIKI_REPO := CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary

.DEFAULT_GOAL := help
.PHONY: help download-wiki full winner serve

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo "  Benchmarks: make -C benchmark"

download-wiki: ## Download the English Wikipedia passages + int8/binary vectors (58 GB, resumable)
	hf download $(WIKI_REPO) --repo-type dataset --include "en/*" --local-dir data/raw

full: ## Write the full corpus (41.5M) as aligned int8, binary, text.bin + offsets into data/slices/full
	uv run scripts/make_full.py

winner: ## Build the winning setup's two indexes for the full corpus (~1.75 h, peak ~23 GB RAM; needs make full)
	uv run scripts/build_winner.py

serve: ## Search page over the full corpus with the winning setup, at http://127.0.0.1:8000 (needs make winner and COHERE_KEY in .env)
	uv run --env-file .env scripts/serve.py
