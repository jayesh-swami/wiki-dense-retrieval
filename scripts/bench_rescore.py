# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu", "numpy", "usearch"]
# ///
# Time each stage of "binary scan + int8 rescore" separately, one thread,
# on the first 500 val questions: FAISS binary scan, gathering the candidate
# int8 rows, and scoring them with numpy or with USearch.
# Adds rows to results/rescore_stages.csv (slice, depth, stage, p50_us, p99_us).
# Usage: bench_rescore.py 5m 300,3000,30000

import csv
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from usearch.index import MetricKind, search

slice_name, depths = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
d = Path("data/slices") / slice_name
faiss.omp_set_num_threads(1)

X = np.load(d / "int8.npy")
B = np.load(d / "binary.npy")
Q8 = np.load("data/questions/val/q_int8.npy")[:500]
Qb = np.load("data/questions/val/q_binary.npy")[:500]
index = faiss.IndexBinaryFlat(1024)
index.add(B)


def timed(f):
    t = time.perf_counter_ns()
    out = f()
    return out, (time.perf_counter_ns() - t) / 1e3


rows = []
for depth in depths:
    times = {"scan": [], "gather": [], "numpy_score": [], "usearch_score": []}
    for q8, qb in zip(Q8, Qb):
        cand, t = timed(lambda: index.search(qb[None], depth)[1][0])
        times["scan"].append(t)
        sub, t = timed(lambda: X[cand])
        times["gather"].append(t)
        _, t = timed(lambda: np.argsort(-(sub.astype(np.int32) @ q8.astype(np.int32)))[:10])
        times["numpy_score"].append(t)
        _, t = timed(lambda: search(sub, q8, 10, MetricKind.IP, exact=True, threads=1).keys)
        times["usearch_score"].append(t)
    for stage, v in times.items():
        p50, p99 = np.percentile(v, [50, 99])
        rows.append({"slice": slice_name, "depth": depth, "stage": stage,
                     "p50_us": f"{p50:.0f}", "p99_us": f"{p99:.0f}"})
        print(f"{slice_name} depth {depth} {stage}: p50 {p50:.0f} us, p99 {p99:.0f} us")

table = Path("results/rescore_stages.csv")
new = not table.exists()
with table.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    if new:
        w.writeheader()
    w.writerows(rows)
