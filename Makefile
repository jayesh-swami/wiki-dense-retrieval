# wiki-hybrid-retrieval Makefile
# Run `make` to see all targets. Add new ones as the project needs them.
# Recipes must be indented with tabs, not spaces.

WIKI_REPO := CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary
NQ_REPO := google-research-datasets/nq_open

.DEFAULT_GOAL := help
.PHONY: help download-wiki download-nq embed-questions

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

download-wiki: ## Download the English Wikipedia passages + int8/binary vectors (58 GB, resumable)
	hf download $(WIKI_REPO) --repo-type dataset --include "en/*" --local-dir data/raw

download-nq: ## Download the NQ-open questions
	hf download $(NQ_REPO) --repo-type dataset --local-dir data/nq_open

embed-questions: ## Embed the NQ-open val and tune questions with Cohere
	uv run --env-file .env scripts/embed_questions.py
