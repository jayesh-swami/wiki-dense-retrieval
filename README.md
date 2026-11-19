# wiki-hybrid-retrieval

Fast retrieval over all of English Wikipedia (41.5M passages) on a single laptop. The goal is the lowest p95 and p99 search latency that still finds the right passages.

Each search setup is compared with exact search on the same vectors. A setup counts only if it passes two checks.

- recall@10 of at least 0.95 against the exact top 10
- answer accuracy within 1 point of exact search (does a top 10 passage contain an answer to the question)

Among the setups that pass, the one with the lowest p99 wins. Settings are tuned on 3,000 NQ-open train questions and reported on the 3,610 validation questions. Each question is searched alone, and the timer covers the whole search, from the question vector to the final 10 passage ids. Embedding the question is not included.

## Result

The winner at 41.5M passages is a union of two approximate indexes, searched in parallel on two threads.

- **Binary HNSW** (USearch, 1-bit vectors, Hamming distance, M=16) keeps its best 250 candidates.
- **IVF-PQ fast-scan** (FAISS, 65,536 cells with an HNSW graph over the centres, 64-byte PQ codes) scans 128 cells and keeps its best 200.

The two candidate lists are merged (about 371 passages per question) and rescored with the exact int8 vectors, which are read from the SSD. The top 10 after the rescore is the answer.

| NQ-open val, 3,610 questions, cold page cache, median of 5 runs | |
|---|---|
| p50 | **2.66 ms** |
| p95 | **3.24 ms** |
| p99 | **3.96 ms** |
| recall@10 | 0.956 |
| answer accuracy@10 | 77.23% (exact search 78.20%, bar 77.20%) |
| RAM for the indexes | 14.8 GB (11.5 GB HNSW + 3.3 GB IVF-PQ) |
| int8 vectors | 40 GB, on the SSD |

About half of each search is the two index lookups and half is reading the ~371 int8 vectors from the SSD. The reads are started together with `posix_fadvise(WILLNEED)` from four threads, and they reach about 75 to 80% of the drive's random-read limit as measured with fio. A cold page cache is the worst case. With real traffic, popular passages would stay cached.

The two indexes miss different passages. Each one alone stays below the recall bar at 41.5M, and together they pass.

**These numbers are for continuous queries.** The benchmark sends the questions one after another with no gap, after 50 warm-up queries. On the search page below, each query first waits a few hundred milliseconds for the Cohere embedding call, and retrieval then takes about 7 ms (median). With a 2 second pause before each query it takes about 11 ms. On this machine the gap between queries most likely lets other work evict the indexes' hot data from the CPU caches and lets the cores go idle, so each search starts cold. A system under steady traffic stays warm.

**Final check.** The two best unions were run 5 times each on val, alternating, because a single run's p99 moves by 1 to 2 ms with the SSD's slow reads. The HNSW side keeping 250 candidates beat keeping 300 at every percentile (median p99 3.96 against 4.51 ms, p50 2.66 against 2.81 ms). Recall and accuracy were the same in every run. Both clear the accuracy bar only narrowly, by 1 and 3 questions out of 3,610.

**Machine.** A laptop with 27 GB of RAM available to the CPU and a WD SN770M NVMe SSD. Only the indexes fit in RAM, so the int8 vectors stay on disk.

## What was tried

### 5M passages (all vectors in RAM)

| Setup | Outcome |
|---|---|
| Exact int8 search | Reference only. Too slow to serve. |
| Binary scan over all passages + int8 rescore | Dropped. Accurate, but scanning every passage is far too slow. |
| HNSW on int8 vectors | Fastest at 5M, but dropped because the index would need about 54 GB at 41.5M. |
| IVF with int8 lists (flat and HNSW coarse search, several cell counts) | Dropped. It needs to scan many cells to reach the recall bar, and the scan is limited by memory bandwidth. |
| IMI (inverted multi-index) | Dropped. Cells are very uneven, so the slow tail is long. |
| BM25 (Tantivy) | Kept as optional. On NQ it rarely finds an answer that dense search misses, and it is much slower. |
| Binary HNSW + int8 rescore | Shortlisted. |
| IVF-PQ fast-scan + int8 rescore | Shortlisted, mainly for its small memory footprint. |
| Union of binary HNSW and IVF-PQ, one rescore | Shortlisted as the lead. Two setups that each miss the bar pass it together. |

### 41.5M passages (int8 vectors on SSD)

| Setup | Outcome |
|---|---|
| Binary HNSW + int8 rescore (two settings) | Pass on the tuning questions, but slower than the union. |
| IVF-PQ fast-scan + int8 rescore (five settings) | All miss the recall bar at this scale. Lowest memory of all (about 7 GB in use). |
| Union of binary HNSW and IVF-PQ (six settings) | **Winner** (above), passing on both the tuning and validation questions. The fastest setting misses the accuracy bar by one question out of 3,610. |

## Status

Work in progress.

- Done: all setups at 5M, the shortlist at 41.5M, the winner confirmed on the validation set.
- Next: an answer step with context engineering on top of the winner. A BM25 recheck at 41.5M is optional.

## Running it

Everything runs through `make`. Run `make` to list the steps (download, build the full corpus files, embed questions, baselines, benchmarks). Scripts are in `scripts/`, and results go to `results/`.

The winner, on the validation questions with a cold cache:

```
uv run scripts/bench.py full union h.M=16 h.ef=250 h.depth=250 i.nlist=65536 i.coarse=hnsw i.nprobe=128 i.depth=200 int8=ssd cache=cold qset=val
```

**Search page.** `make serve` starts a small local page at http://127.0.0.1:8000 over the winning setup. It embeds the query with Cohere (needs `COHERE_KEY` in `.env`), retrieves the top 10 passages and shows the embedding and retrieval times. It loads the indexes in about 20 seconds and holds about 15 GB of RAM while running.

## Data

- **Passages and vectors:** [Cohere's Wikipedia 2023-11 embeddings](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary) (`embed-multilingual-v3.0`, int8 and binary), English config. Thanks to Cohere for publishing them.
- **Wikipedia text:** © Wikipedia contributors, from Wikimedia, under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).
- **Questions:** [NQ-open](https://huggingface.co/datasets/google-research-datasets/nq_open) from Google's Natural Questions, under CC BY-SA. Questions were embedded with the same Cohere model.
