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
#        bench.py 5m binary depth=200 rescore=usearch qset=tune
#        bench.py 5m hnsw M=16 efc=128 ef=64 qset=tune
#        bench.py 5m ivf coarse=flat|hnsw nlist=16384 nprobe=64 qset=tune
#        bench.py 5m ivf coarse=imi bits=10 nprobe=4096 qset=tune

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

from answers import has_answer

slice_name, setup = sys.argv[1], sys.argv[2]
params = dict(a.split("=", 1) for a in sys.argv[3:])
qset = params.pop("qset", "val")
d = Path("data/slices") / slice_name
faiss.omp_set_num_threads(1)


def build_exact(X, B, p):
    # Brute force over every int8 vector (USearch SIMD dot products).
    return lambda q8, qb: search(X, q8, 10, MetricKind.IP, exact=True, threads=1).keys


def build_binary(X, B, p):
    # Hamming scan over all binary vectors (FAISS), keep `depth` candidates,
    # rescore those with int8 dot products (numpy or usearch), return the top 10.
    depth = int(p.get("depth", 200))
    rescore = p.get("rescore", "numpy")
    index = faiss.IndexBinaryFlat(1024)
    index.add(B)

    def run(q8, qb):
        cand = index.search(qb[None], depth)[1][0]
        if rescore == "usearch":
            return cand[search(X[cand], q8, 10, MetricKind.IP, exact=True, threads=1).keys]
        scores = X[cand].astype(np.int32) @ q8.astype(np.int32)
        return cand[np.argsort(-scores)[:10]]
    return run


def build_hnsw(X, B, p):
    # USearch HNSW on the int8 vectors. The graph walk scores with int8 dot
    # products, so no rescore. The built index is saved and reused, because
    # only ef (search effort) changes between runs; the build uses all cores.
    M, efc, ef = int(p.get("M", 16)), int(p.get("efc", 128)), int(p.get("ef", 64))
    path = Path(f"data/indexes/{slice_name}/hnsw_i8_M{M}_efc{efc}.usearch")
    index = Index(ndim=1024, metric="ip", dtype="i8", connectivity=M, expansion_add=efc)
    if path.exists():
        index.load(path)
    else:
        t = time.perf_counter()
        index.add(np.arange(len(X)), X, threads=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        index.save(path)
        path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
    index.expansion_search = ef
    p["build_s"] = path.with_suffix(".build_s").read_text()
    return lambda q8, qb: index.search(q8, 10, threads=1).keys


def build_ivf(X, B, p):
    # FAISS IVF on the int8 vectors. The lists store the vectors as exact int8
    # (SQ8_direct_signed, no residuals), so the scan is an exact int8 dot product.
    # coarse=flat: brute force over nlist centres to pick the nprobe nearest.
    # coarse=hnsw: same trained index, but an HNSW graph over the centres picks them.
    # coarse=imi:  inverted multi-index, 2^bits x 2^bits cells from the two vector
    #              halves. FAISS only routes IMI by L2; the lists are still scored by IP.
    # max_codes caps how many vectors are scanned (0 = no cap).
    coarse = p.get("coarse", "flat")
    nprobe, max_codes = int(p.get("nprobe", 32)), int(p.get("max_codes", 0))
    name = f"IMI2x{p.get('bits', 10)}" if coarse == "imi" else f"IVF{p.get('nlist', 16384)}"
    path = Path(f"data/indexes/{slice_name}/{name}_SQ8s.faiss")
    if path.exists():
        index = faiss.read_index(str(path))
    else:
        if coarse == "imi":
            bits = int(p.get("bits", 10))
            q = faiss.MultiIndexQuantizer(1024, 2, bits)
            index = faiss.IndexIVFScalarQuantizer(q, 1024, 4 ** bits, faiss.ScalarQuantizer.QT_8bit_direct_signed,
                                                  faiss.METRIC_INNER_PRODUCT, False)
            index.quantizer_trains_alone = 1
            index.own_fields = True
            q.this.disown()
        else:
            index = faiss.index_factory(1024, f"{name},SQ8_direct_signed", faiss.METRIC_INNER_PRODUCT)
            index.by_residual = False
        faiss.omp_set_num_threads(os.cpu_count())
        t = time.perf_counter()
        n = 640_000 if coarse == "imi" else 39 * index.nlist  # 39 vectors per centre (FAISS minimum)
        sample = np.sort(np.random.default_rng(0).choice(len(X), n, replace=False))
        index.train(X[sample].astype(np.float32))
        for i in range(0, len(X), 200_000):
            index.add(X[i:i + 200_000].astype(np.float32))
        faiss.omp_set_num_threads(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
        # list sizes, to see how uneven the cells are
        sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)])
        np.save(path.with_suffix(".sizes.npy"), sizes)
        nz = sizes[sizes > 0]
        print(f"{name}: {len(nz)} of {len(sizes)} cells non-empty, mean {nz.mean():.0f}, "
              f"p99 {np.percentile(nz, 99):.0f}, max {nz.max()}")
    if coarse == "hnsw":
        # HNSW over the trained centres (seconds to build at 16k centres)
        hq = faiss.IndexHNSWFlat(1024, 32, faiss.METRIC_INNER_PRODUCT)
        hq.add(index.quantizer.reconstruct_n(0, index.nlist))
        hq.hnsw.efSearch = max(64, 2 * nprobe)
        index.quantizer = hq
        index.own_fields = False
    index.nprobe, index.max_codes = nprobe, max_codes
    p["build_s"] = path.with_suffix(".build_s").read_text()
    run = lambda q8, qb: index.search(q8[None].astype(np.float32), 10)[1][0]
    run.keep = index.quantizer if coarse != "hnsw" else hq  # keep the quantizer alive
    return run


SETUPS = {"exact": build_exact, "binary": build_binary, "hnsw": build_hnsw, "ivf": build_ivf}

# Load data
X = np.load(d / "int8.npy", mmap_mode="r")
B = np.load(d / "binary.npy")
Q8 = np.load(f"data/questions/{qset}/q_int8.npy")
Qb = np.load(f"data/questions/{qset}/q_binary.npy")
warm = "tune" if qset == "val" else "val"
warm8 = np.load(f"data/questions/{warm}/q_int8.npy")[:50]
warmb = np.load(f"data/questions/{warm}/q_binary.npy")[:50]
gt = np.load(f"baselines/{slice_name}/top-100/gt_{qset}.npy")[:, :10]
questions = pq.read_table(f"data/questions/{qset}/questions.parquet")
answers = questions["answers" if "answers" in questions.column_names else "answer"].to_pylist()
text = pq.read_table(d / "passages.parquet", columns=["text"])["text"]

# Build
t = time.perf_counter()
run_search = SETUPS[setup](X, B, params)
build_s = float(params.pop("build_s", time.perf_counter() - t))  # hnsw reports its saved build time

# Search
for q8, qb in zip(warm8, warmb):
    run_search(q8, qb)
found = np.zeros((len(Q8), 10), dtype=np.int64)
lat_ns = np.zeros(len(Q8), dtype=np.int64)
for i, (q8, qb) in enumerate(zip(Q8, Qb)):
    t = time.perf_counter_ns()
    keys = run_search(q8, qb)
    lat_ns[i] = time.perf_counter_ns() - t
    found[i] = keys

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
