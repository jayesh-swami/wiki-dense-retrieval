# wiki-hybrid-retrieval

Fast retrieval over all of English Wikipedia (41.5M passages) on a single laptop. The goal is the lowest p95 and p99 search latency that still finds the right passages.

Each search setup is compared with exact search on the same vectors. A setup counts only if it passes two checks:

- recall@10 of at least 0.95 against the exact top 10
- answer accuracy within 1 point of exact search (does a top 10 passage contain an answer to the question)

Among the setups that pass, the one with the lowest p99 wins. Settings are tuned on 3,000 NQ-open train questions and reported once on the 3,610 validation questions. Each question is searched alone, on one thread.

## Status

Work in progress.

- Done on a 5M passage sample: exact search, binary scan with int8 rescore, HNSW, IVF and IMI.
- Next: compressed indexes, BM25 + dense hybrids, the full 41.5M corpus, and an answer step with context engineering.

## Running it

Everything runs through `make`. Run `make` to list the steps (download, embed questions, slice, baselines, benchmarks). Scripts are in `scripts/`, and results go to `results/`.

## Data

- **Passages and vectors:** [Cohere's Wikipedia 2023-11 embeddings](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary) (`embed-multilingual-v3.0`, int8 and binary), English config. Thanks to Cohere for publishing them.
- **Wikipedia text:** © Wikipedia contributors, from Wikimedia, under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
- **Questions:** [NQ-open](https://huggingface.co/datasets/google-research-datasets/nq_open) from Google's Natural Questions, under CC BY-SA. Questions were embedded with the same Cohere model.
