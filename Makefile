# wiki-hybrid-retrieval Makefile
# Run `make` to see all targets. Add new ones as the project needs them.
# Recipes must be indented with tabs, not spaces.

WIKI_REPO := CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary
NQ_REPO := google-research-datasets/nq_open

# make slice K=24 SLICE=1m: keep articles with article_id % 1000 < K
K ?= 120
# slice every other target works on
SLICE ?= 5m
# how many exact results to keep per question
TOPK ?= 100
# make bench SETUP=binary PARAMS="depth=200" QSET=tune
SETUP ?= exact
PARAMS ?=
QSET ?= val
# rescore depths for make bench-rescore
DEPTHS ?= 300,3000,30000
# accuracy@N values to report, comma separated (each must be <= TOPK)
AT ?= 10,20,100

.DEFAULT_GOAL := help
.PHONY: help download-wiki download-nq embed-questions slice full baseline baseline-topk-exact baseline-accuracy baseline-bench bench bench-rescore

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

download-wiki: ## Download the English Wikipedia passages + int8/binary vectors (58 GB, resumable)
	hf download $(WIKI_REPO) --repo-type dataset --include "en/*" --local-dir data/raw

download-nq: ## Download the NQ-open questions
	hf download $(NQ_REPO) --repo-type dataset --local-dir data/nq_open

embed-questions: ## Embed the NQ-open val and tune questions with Cohere
	uv run --env-file .env scripts/embed_questions.py

slice: ## Sample whole articles (article_id % 1000 < K) into data/slices/SLICE (K=120 SLICE=5m)
	uv run scripts/make_slice.py $(K) $(SLICE)

full: ## Write the full corpus (41.5M) as aligned int8, binary, text.bin + offsets into data/slices/full
	uv run scripts/make_full.py

baseline: baseline-topk-exact baseline-accuracy baseline-bench ## Run all baselines (SLICE=5m TOPK=100 AT=10,20,100)

baseline-topk-exact: ## Exact top TOPK passages per question into baselines/SLICE/top-TOPK
	uv run scripts/baseline_topk_exact.py $(SLICE) $(TOPK)

baseline-accuracy: ## Answer accuracy@AT of the exact top TOPK on val, into baselines/accuracy.csv
	uv run scripts/baseline_accuracy.py $(SLICE) $(TOPK) $(AT)

baseline-bench: ## Time exact brute-force search on val, one question at a time (needs top-100); into results/
	uv run scripts/bench.py $(SLICE) exact

bench: ## Benchmark SETUP (exact|binary|hnsw|hnsw_binary|ivf|bm25|union) with PARAMS on QSET (val|tune); into results/
	uv run scripts/bench.py $(SLICE) $(SETUP) $(PARAMS) qset=$(QSET)

bench-rescore: ## Time each stage of binary scan + int8 rescore at DEPTHS; into results/rescore_stages.csv
	uv run scripts/bench_rescore.py $(SLICE) $(DEPTHS)
