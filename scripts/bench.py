# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu", "numpy", "pyarrow", "usearch"]
# ///
# Benchmark one search setup on one slice: build it, run every question of a set
# (val or tune) one at a time on one thread, and score recall@10 and answer accuracy@10.
#
# Timing: time.perf_counter_ns() around the search call only, per question,
# after 50 untimed warm-up searches. Question loading and scoring are outside
# the timer. p50/p95/p99 come from those per-question times.
#
# Appends one row to results/results.csv and saves the raw per-question times
# to results/latencies/<run>.npy.
# Usage: bench.py 5m exact
#        bench.py 5m binary depth=300 qset=tune
#        bench.py 5m hnsw M=32 efc=128 ef=128 qset=tune
#        bench.py 5m hnsw_binary M=32 efc=128 ef=256 depth=100 qset=tune
#        bench.py 5m ivf coarse=flat|hnsw nlist=16384 nprobe=384 qset=tune

import csv
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from usearch.index import Index, MetricKind, search

from answers import has_answer, load_answers

slice_name, setup = sys.argv[1], sys.argv[2]
params = dict(a.split("=", 1) for a in sys.argv[3:])
qset = params.pop("qset", "val")
d = Path("data/slices") / slice_name
faiss.omp_set_num_threads(1)

# Each setup returns (search function, build seconds). Saved indexes return the
# build time recorded when they were built; None means "time the setup call".


def exact_top10(X, q8):
    # Brute force int8 dot products (USearch SIMD), one thread.
    return search(X, q8, 10, MetricKind.IP, exact=True, threads=1).keys


def build_exact(X, B, p):
    return lambda q8, qb: exact_top10(X, q8), None


def build_binary(X, B, p):
    # Hamming scan over all binary vectors (FAISS), keep `depth` candidates,
    # rescore those with exact int8 dot products, return the top 10.
    depth = int(p.get("depth", 300))
    index = faiss.IndexBinaryFlat(1024)
    index.add(B)

    def run(q8, qb):
        cand = index.search(qb[None], depth)[1][0]
        return cand[exact_top10(X[cand], q8)]
    return run, None


def load_or_build(index, path, vectors):
    # Load a saved USearch index, or build it on all cores and save it with its
    # build time. Indexes are reused because only ef/depth change between runs.
    if path.exists():
        index.load(path)
    else:
        t = time.perf_counter()
        index.add(np.arange(len(vectors)), vectors, threads=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        index.save(path)
        path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
    return float(path.with_suffix(".build_s").read_text())


def build_hnsw(X, B, p):
    # USearch HNSW on the int8 vectors. The graph walk scores with int8 dot
    # products, so no rescore.
    M, efc, ef = int(p.get("M", 32)), int(p.get("efc", 128)), int(p.get("ef", 128))
    index = Index(ndim=1024, metric="ip", dtype="i8", connectivity=M, expansion_add=efc)
    build_s = load_or_build(index, Path(f"data/indexes/{slice_name}/hnsw_i8_M{M}_efc{efc}.usearch"), X)
    index.expansion_search = ef
    return lambda q8, qb: index.search(q8, 10, threads=1).keys, build_s


def build_hnsw_binary(X, B, p):
    # USearch HNSW on the binary vectors (Hamming distance), keep `depth`
    # candidates, rescore those with exact int8 dot products, return the top 10.
    # The int8 vectors are read into RAM so the page cache can't change results.
    # USearch searches with max(ef, depth), so ef below depth has no effect.
    M, efc = int(p.get("M", 32)), int(p.get("efc", 128))
    ef, depth = int(p.get("ef", 256)), int(p.get("depth", 100))
    X = np.array(X)
    index = Index(ndim=1024, metric="hamming", dtype="b1", connectivity=M, expansion_add=efc)
    build_s = load_or_build(index, Path(f"data/indexes/{slice_name}/hnsw_b1_M{M}_efc{efc}.usearch"), B)
    index.expansion_search = ef

    def run(q8, qb):
        cand = index.search(qb, depth, threads=1).keys
        return cand[exact_top10(X[cand], q8)]
    return run, build_s


def build_ivf(X, B, p):
    # FAISS IVF on the int8 vectors. The lists store the vectors as exact int8
    # (SQ8_direct_signed, no residuals), so the scan is an exact int8 dot product.
    # coarse=flat: brute force over the nlist centres to pick the nprobe nearest.
    # coarse=hnsw: same saved index, but an HNSW graph over the centres picks them.
    # (IMI was tried on 18 Nov 2026 and dropped; its code is in commit 9d88fc9.)
    nlist, nprobe = int(p.get("nlist", 16384)), int(p.get("nprobe", 384))
    path = Path(f"data/indexes/{slice_name}/IVF{nlist}_SQ8s.faiss")
    if path.exists():
        index = faiss.read_index(str(path))
    else:
        index = faiss.index_factory(1024, f"IVF{nlist},SQ8_direct_signed", faiss.METRIC_INNER_PRODUCT)
        index.by_residual = False
        faiss.omp_set_num_threads(os.cpu_count())
        t = time.perf_counter()
        sample = np.sort(np.random.default_rng(0).choice(len(X), 39 * nlist, replace=False))  # FAISS minimum
        index.train(X[sample].astype(np.float32))
        for i in range(0, len(X), 200_000):
            index.add(X[i:i + 200_000].astype(np.float32))
        faiss.omp_set_num_threads(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
        sizes = np.array([index.invlists.list_size(i) for i in range(nlist)])
        print(f"IVF{nlist} list sizes: mean {sizes.mean():.0f}, p99 {np.percentile(sizes, 99):.0f}, max {sizes.max()}")
    if p.get("coarse", "flat") == "hnsw":
        hq = faiss.IndexHNSWFlat(1024, 32, faiss.METRIC_INNER_PRODUCT)
        hq.add(index.quantizer.reconstruct_n(0, nlist))
        hq.hnsw.efSearch = max(64, 2 * nprobe)
        index.quantizer, index.own_fields = hq, False
        index.referenced_objects = [hq]  # keeps hq alive in Python; FAISS doesn't own it
    index.nprobe = nprobe
    return (lambda q8, qb: index.search(q8[None].astype(np.float32), 10)[1][0],
            float(path.with_suffix(".build_s").read_text()))


SETUPS = {"exact": build_exact, "binary": build_binary, "hnsw": build_hnsw,
          "hnsw_binary": build_hnsw_binary, "ivf": build_ivf}

# Load data
X = np.load(d / "int8.npy", mmap_mode="r")
B = np.load(d / "binary.npy")
Q8 = np.load(f"data/questions/{qset}/q_int8.npy")
Qb = np.load(f"data/questions/{qset}/q_binary.npy")
warm = "tune" if qset == "val" else "val"
warm8 = np.load(f"data/questions/{warm}/q_int8.npy")[:50]
warmb = np.load(f"data/questions/{warm}/q_binary.npy")[:50]
gt = np.load(f"baselines/{slice_name}/top-100/gt_{qset}.npy")[:, :10]
answers = load_answers(qset)
text = pq.read_table(d / "passages.parquet", columns=["text"])["text"]

# Build
t = time.perf_counter()
run_search, build_s = SETUPS[setup](X, B, params)
build_s = build_s or time.perf_counter() - t

# Search
for q8, qb in zip(warm8, warmb):
    run_search(q8, qb)
found = np.zeros((len(Q8), 10), dtype=np.int64)
lat_ns = np.zeros(len(Q8), dtype=np.int64)
for i, (q8, qb) in enumerate(zip(Q8, Qb)):
    t = time.perf_counter_ns()
    found[i] = run_search(q8, qb)
    lat_ns[i] = time.perf_counter_ns() - t

# Score
recall = np.mean([len(set(f) & set(g)) / 10 for f, g in zip(found, gt)])
passages = text.take(found.ravel()).to_pylist()
accuracy = np.mean([any(has_answer(p, ans) for p in passages[i * 10:(i + 1) * 10])
                    for i, ans in enumerate(answers)])
ms = lat_ns / 1e6
p50, p95, p99 = np.percentile(ms, [50, 95, 99])

# Save
tag = "_".join(f"{k}{v}" for k, v in params.items())
run = f"{datetime.now():%Y%m%d-%H%M%S}_{slice_name}_{setup}" + (f"_{tag}" if tag else "") + f"_{qset}"
Path("results/latencies").mkdir(parents=True, exist_ok=True)
np.save(f"results/latencies/{run}.npy", lat_ns)
row = {
    "run": run, "slice": slice_name, "setup": setup, "params": json.dumps(params),
    "qset": qset, "questions": len(Q8),
    "recall@10": f"{recall:.4f}", "accuracy@10": f"{accuracy:.4f}",
    "p50_ms": f"{p50:.2f}", "p95_ms": f"{p95:.2f}", "p99_ms": f"{p99:.2f}",
    "mean_ms": f"{ms.mean():.2f}", "build_s": f"{build_s:.1f}",
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
}
table = Path("results/results.csv")
new = not table.exists()
with table.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if new:
        w.writeheader()
    w.writerow(row)
print(row)
