# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu", "numpy", "usearch"]
# ///
# Build the two indexes of the winning setup for the full corpus (data/slices/full),
# one after the other, skipping any that already exist:
#   1. binary HNSW (USearch, M=16, efc=128) over binary.npy
#      ~13 min, peak ~17 GiB RAM -> data/indexes/full/hnsw_b1_M16_efc128.usearch (11.5 GB)
#   2. IVF65536 + PQ128x4fs (FAISS fast-scan, 64 B per passage, inner product) over int8.npy,
#      trained on a 1.3M-passage sample (FAISS needs float32), then added in batches of 1M.
#      Passages are assigned to cells through an HNSW graph over the centres.
#      ~1.4 h (1 h of it k-means), peak ~22 GiB RAM -> data/indexes/full/IVF65536_PQ128x4fs.faiss (3.3 GB)
# Each index gets a .build_s file with its build time, which the benchmark reads.
# Usage: build_winner.py

import os
import time
from pathlib import Path

import faiss
import numpy as np
from usearch.index import Index

D, OUT = Path("data/slices/full"), Path("data/indexes/full")
OUT.mkdir(parents=True, exist_ok=True)

path = OUT / "hnsw_b1_M16_efc128.usearch"
if path.exists():
    print(f"{path} exists, skipping")
else:
    B = np.load(D / "binary.npy", mmap_mode="r")
    t = time.perf_counter()
    index = Index(ndim=1024, metric="hamming", dtype="b1", connectivity=16, expansion_add=128)
    index.add(np.arange(len(B)), B, threads=0)
    index.save(path)
    path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
    print(f"built {path} in {time.perf_counter() - t:.0f} s", flush=True)
    del index

path = OUT / "IVF65536_PQ128x4fs.faiss"
if path.exists():
    print(f"{path} exists, skipping")
else:
    X = np.load(D / "int8.npy", mmap_mode="r")
    faiss.omp_set_num_threads(os.cpu_count())
    index = faiss.index_factory(1024, "IVF65536_HNSW32,PQ128x4fs", faiss.METRIC_INNER_PRODUCT)
    t = time.perf_counter()
    sample = np.sort(np.random.default_rng(0).choice(len(X), 1_300_000, replace=False))
    index.train(X[sample].astype(np.float32))
    del sample
    print(f"trained in {time.perf_counter() - t:.0f} s", flush=True)
    faiss.downcast_index(index.quantizer).hnsw.efSearch = 64  # cell assignment quality
    for i in range(0, len(X), 1_000_000):
        index.add(X[i:i + 1_000_000].astype(np.float32))
        print(f"added {min(i + 1_000_000, len(X)):,}", flush=True)
    faiss.write_index(index, str(path))
    path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
    print(f"built {path} in {time.perf_counter() - t:.0f} s", flush=True)
