# Benchmarks. Run `make` here (or `make -C benchmark` from the repo root) to see all targets.
# Scripts run from the repo root, so data/ paths work. Results go to benchmark/results/,
# exact ground truth to benchmark/baselines/ (not in git).
# Recipes must be indented with tabs, not spaces.

ROOT := ..
# slice every target works on (5m, 1m, full)
SLICE ?= 5m
# how many exact results to keep per question
TOPK ?= 100
# accuracy@N values to report, comma separated (each must be <= TOPK)
AT ?= 10,20,100
# make bench SETUP=binary PARAMS="depth=200" QSET=tune
SETUP ?= exact
PARAMS ?=
QSET ?= val
# rescore depths for make bench-rescore
DEPTHS ?= 300,3000,30000
# the winning setup (see the README)
WINNER := h.M=16 h.ef=250 h.depth=250 i.nlist=65536 i.coarse=hnsw i.nprobe=128 i.depth=200 int8=ssd cache=cold

.DEFAULT_GOAL := help
.PHONY: help baseline baseline-topk-exact baseline-accuracy baseline-bench bench bench-rescore bench-winner

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

baseline: baseline-topk-exact baseline-accuracy baseline-bench ## Run all baselines (SLICE=5m TOPK=100 AT=10,20,100)

baseline-topk-exact: ## Exact top TOPK passages per question into benchmark/baselines/SLICE/top-TOPK
	cd $(ROOT) && uv run benchmark/scripts/baseline_topk_exact.py $(SLICE) $(TOPK)

baseline-accuracy: ## Answer accuracy@AT of the exact top TOPK on val, into benchmark/baselines/accuracy.csv
	cd $(ROOT) && uv run benchmark/scripts/baseline_accuracy.py $(SLICE) $(TOPK) $(AT)

baseline-bench: ## Time exact brute-force search on val, one question at a time (needs top-100); into results/
	cd $(ROOT) && uv run benchmark/scripts/bench.py $(SLICE) exact

bench: ## Benchmark SETUP (exact|binary|hnsw|hnsw_binary|ivf|bm25|union) with PARAMS on QSET (val|tune); into results/
	cd $(ROOT) && uv run benchmark/scripts/bench.py $(SLICE) $(SETUP) $(PARAMS) qset=$(QSET)

bench-winner: ## Benchmark the winning setup on the full corpus, QSET=val, cold cache (needs make winner + baselines)
	cd $(ROOT) && uv run benchmark/scripts/bench.py full union $(WINNER) qset=$(QSET)

bench-rescore: ## Time each stage of binary scan + int8 rescore at DEPTHS; into results/rescore_stages.csv
	cd $(ROOT) && uv run benchmark/scripts/bench_rescore.py $(SLICE) $(DEPTHS)
