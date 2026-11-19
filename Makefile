# wiki-dense-retrieval Makefile
# Run `make` to see all targets. Benchmarks have their own Makefile: `make -C benchmark`.
# Recipes must be indented with tabs, not spaces.

WIKI_REPO := CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary

.DEFAULT_GOAL := help
.PHONY: help download-wiki full winner web serve web-dev

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo "  Benchmarks: make -C benchmark"

download-wiki: ## Download the English Wikipedia passages + int8/binary vectors (58 GB, resumable)
	hf download $(WIKI_REPO) --repo-type dataset --include "en/*" --local-dir data/raw

full: ## Write the full corpus (41.5M) as aligned int8, binary, text.bin + offsets into data/slices/full
	uv run scripts/make_full.py

winner: ## Build the winning setup's two indexes for the full corpus (~1.6 h, peak ~22 GiB RAM; needs make full)
	uv run scripts/build_winner.py

web: ## Build the React search page into web/dist
	cd web && npm install && npm run build

serve: web ## Search API + page at http://127.0.0.1:8000 (needs make winner and COHERE_KEY in .env; ~18 GiB RAM)
	uv run --env-file .env api/main.py

web-dev: ## React dev server at http://127.0.0.1:5173, forwarding /api to make serve
	cd web && npm install && npm run dev
