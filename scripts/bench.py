# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu", "numpy", "pyarrow", "tantivy", "usearch"]
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
#        bench.py 5m ivf lists=PQ128x4fs nprobe=128 depth=300 qset=tune
#        bench.py 5m bm25 qset=tune
#        bench.py full hnsw_binary M=16 ef=600 depth=300 int8=ssd cache=cold qset=tune
#        bench.py 5m union h.M=16 h.ef=300 h.depth=300 i.coarse=hnsw i.nprobe=128 i.depth=300 qset=tune

import csv
import functools
import json
import os
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import tantivy
from usearch.index import Index, MetricKind, search

from answers import has_answer, load_answers, passage_texts

slice_name, setup = sys.argv[1], sys.argv[2]
params = dict(a.split("=", 1) for a in sys.argv[3:])
qset = params.pop("qset", "val")
K = int(params.get("k", 10))  # passages returned per question (10 or 20)
d = Path("data/slices") / slice_name
faiss.omp_set_num_threads(1)

# Each setup returns (search function, build seconds). The search function takes
# the question's int8 vector, binary vector and text, and returns 10 passage ids.
# Saved indexes return the build time recorded when they were built; None means
# "time the setup call". Setups that rescore also attach run.cands, which returns
# the candidates before the rescore (used by the union setup).


def exact_top(X, q8):
    # Brute force int8 dot products (USearch SIMD), one thread; best K, best first.
    return search(X, q8, K, MetricKind.IP, exact=True, threads=1).keys


def build_exact(X, B, p):
    return lambda q8, qb, qt: exact_top(X, q8), None


def build_binary(X, B, p):
    # Hamming scan over all binary vectors (FAISS), keep `depth` candidates,
    # rescore those with exact int8 dot products, return the top K.
    depth = int(p.get("depth", 300))
    index = faiss.IndexBinaryFlat(1024)
    index.add(B)

    def run(q8, qb, qt):
        cand = index.search(qb[None], depth)[1][0]
        return cand[exact_top(rows(cand), q8)]
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
    return lambda q8, qb, qt: index.search(q8, K, threads=1).keys, build_s


def build_hnsw_binary(X, B, p):
    # USearch HNSW on the binary vectors (Hamming distance), keep `depth`
    # candidates, rescore those with exact int8 dot products, return the top K.
    # USearch searches with max(ef, depth), so ef below depth has no effect.
    M, efc = int(p.get("M", 32)), int(p.get("efc", 128))
    ef, depth = int(p.get("ef", 256)), int(p.get("depth", 100))
    index = Index(ndim=1024, metric="hamming", dtype="b1", connectivity=M, expansion_add=efc)
    build_s = load_or_build(index, Path(f"data/indexes/{slice_name}/hnsw_b1_M{M}_efc{efc}.usearch"), B)
    index.expansion_search = ef

    def cands(q8, qb, qt):
        return index.search(qb, depth, threads=1).keys.astype(np.int64)

    def run(q8, qb, qt):
        cand = cands(q8, qb, qt)
        return cand[exact_top(rows(cand), q8)]
    run.cands = cands
    return run, build_s


def build_ivf(X, B, p):
    # FAISS IVF on the int8 vectors. The lists store the vectors as exact int8
    # (SQ8_direct_signed, no residuals), so the scan is an exact int8 dot product.
    # coarse=flat: brute force over the nlist centres to pick the nprobe nearest.
    # coarse=hnsw: same saved index, but an HNSW graph over the centres picks them.
    # lists=PQ64 / PQ128x4fs: the lists hold 64-byte PQ codes of the residuals
    # instead; the top `depth` are rescored with exact int8. They reuse the centres
    # of the SQ8s index, so only the list format differs.
    # (IMI was tried on 18 Nov 2026 and dropped; its code is in commit 9d88fc9.)
    nlist, nprobe = int(p.get("nlist", 16384)), int(p.get("nprobe", 384))
    lists, depth = p.get("lists", "SQ8s"), int(p.get("depth", 300))
    path = Path(f"data/indexes/{slice_name}/IVF{nlist}_{lists}.faiss")
    if path.exists():
        index = faiss.read_index(str(path))
    else:
        if lists == "SQ8s":
            index = faiss.index_factory(1024, f"IVF{nlist},SQ8_direct_signed", faiss.METRIC_INNER_PRODUCT)
            index.by_residual = False
        else:
            index = faiss.index_factory(1024, f"IVF{nlist},{lists}", faiss.METRIC_INNER_PRODUCT)
            sq8 = faiss.read_index(str(path.with_name(f"IVF{nlist}_SQ8s.faiss")), faiss.IO_FLAG_MMAP)
            index.quantizer.add(sq8.quantizer.reconstruct_n(0, nlist))  # FAISS then skips coarse training
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
    build_s = float(path.with_suffix(".build_s").read_text())
    if lists == "SQ8s":
        return lambda q8, qb, qt: index.search(q8[None].astype(np.float32), K)[1][0], build_s

    def cands(q8, qb, qt):
        return index.search(q8[None].astype(np.float32), depth)[1][0]

    def run(q8, qb, qt):
        cand = cands(q8, qb, qt)
        return cand[exact_top(rows(cand), q8)]
    run.cands = cands
    return run, build_s


def build_bm25(X, B, p):
    # Tantivy BM25 over title + text. The analyzer is Tantivy's en_stem plus its
    # built-in English stop word list, used for both passages and questions. No
    # positions are indexed (BM25 never reads them). A question is lowercased and
    # stripped of punctuation, then searched as an OR of its words.
    # count=False lets Tantivy skip passages that can't reach the top 10.
    path = Path(f"data/indexes/{slice_name}/bm25")
    analyzer = (tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
                .filter(tantivy.Filter.remove_long(40)).filter(tantivy.Filter.lowercase())
                .filter(tantivy.Filter.stopword("english")).filter(tantivy.Filter.stemmer("english")).build())
    if not path.exists():
        sb = tantivy.SchemaBuilder()
        sb.add_integer_field("id", stored=False, indexed=False, fast=True)
        sb.add_text_field("body", stored=False, tokenizer_name="en_stop", index_option="freq")
        path.mkdir(parents=True)
        index = tantivy.Index(sb.build(), path=str(path))
        index.register_tokenizer("en_stop", analyzer)
        writer = index.writer(4_000_000_000, os.cpu_count())
        t = time.perf_counter()
        i = 0
        for batch in pq.ParquetFile(d / "passages.parquet").iter_batches(columns=["title", "text"]):
            for title, body in zip(batch["title"].to_pylist(), batch["text"].to_pylist()):
                writer.add_document(tantivy.Document(id=i, body=f"{title} {body}"))
                i += 1
        writer.commit()
        writer.wait_merging_threads()
        (path / "build_s").write_text(f"{time.perf_counter() - t:.1f}")
    index = tantivy.Index.open(str(path))
    index.register_tokenizer("en_stop", analyzer)  # not saved with the index
    searcher = index.searcher()

    def run(q8, qb, qt):
        query = index.parse_query(re.sub(r"[^\w\s]", " ", qt.lower()), ["body"])
        hits = searcher.search(query, K, count=False).hits
        return searcher.fast_field_values("id", [a for _, a in hits])
    return run, float((path / "build_s").read_text())


def build_union(X, B, p):
    # Binary HNSW and IVF-PQ side by side: IVF-PQ runs on a second thread while
    # the binary HNSW walk runs on this one (both release the GIL). Their
    # candidates are merged and rescored once with exact int8.
    # Params: h.* go to binary HNSW (M, ef, depth), i.* to IVF-PQ (coarse, nprobe, depth).
    h, hb = build_hnsw_binary(X, B, {k[2:]: v for k, v in p.items() if k.startswith("h.")})
    i, ib = build_ivf(X, B, {"lists": "PQ128x4fs"} | {k[2:]: v for k, v in p.items() if k.startswith("i.")})
    pool = ThreadPoolExecutor(1)
    sizes = []  # merged candidate counts (= SSD reads per question at scale)

    def run(q8, qb, qt):
        ivf = pool.submit(i.cands, q8, qb, qt)
        cand = np.union1d(h.cands(q8, qb, qt), ivf.result())
        sizes.append(len(cand))
        return cand[exact_top(rows(cand), q8)]
    run.sizes = sizes
    return run, hb + ib


SETUPS = {"exact": build_exact, "binary": build_binary, "hnsw": build_hnsw,
          "hnsw_binary": build_hnsw_binary, "ivf": build_ivf, "bm25": build_bm25,
          "union": build_union}

# Load data
X = np.load(d / "int8.npy", mmap_mode="r")
B = np.load(d / "binary.npy", mmap_mode="r")
Q8 = np.load(f"data/questions/{qset}/q_int8.npy")
Qb = np.load(f"data/questions/{qset}/q_binary.npy")
warm = "tune" if qset == "val" else "val"
warm8 = np.load(f"data/questions/{warm}/q_int8.npy")[:50]
warmb = np.load(f"data/questions/{warm}/q_binary.npy")[:50]
gt = np.load(f"baselines/{slice_name}/top-100/gt_{qset}.npy")[:, :10]
answers = load_answers(qset)
texts = pq.read_table(f"data/questions/{qset}/questions.parquet")["question"].to_pylist()
warm_texts = pq.read_table(f"data/questions/{warm}/questions.parquet")["question"].to_pylist()[:50]

# Rescore reads. int8=ram (default): all int8 vectors are copied into RAM on first
# use (fine at 5m). int8=ssd: rows are gathered from the memory-mapped int8.npy,
# after one posix_fadvise(WILLNEED) per row. That queues all reads at once, so the
# SSD serves them in parallel (~1.4 ms for ~400 random rows cold, about this SSD's
# limit), and the gather itself is one NumPy call. cache=cold evicts int8.npy from the page cache after the warm-up,
# so timed reads start on the SSD.
fd = os.open(d / "int8.npy", os.O_RDONLY)


@functools.cache
def x_in_ram():
    return np.array(X)


def rows(ids):
    if params.get("int8", "ram") == "ram":
        return x_in_ram()[ids]
    offs = (ids * 1024 + X.offset).tolist()
    list(advisers.map(advise, [offs[k::4] for k in range(4)]))
    return np.array(X[ids])


# Each cold fadvise costs ~3 us in the kernel (page allocation + read submission), so the
# calls are split over 4 threads (os.posix_fadvise releases the GIL). 8 threads is worse at p99.
advisers = ThreadPoolExecutor(4)


def advise(offs):
    for o in offs:
        os.posix_fadvise(fd, o, 1024, os.POSIX_FADV_WILLNEED)


# Build
t = time.perf_counter()
run_search, build_s = SETUPS[setup](X, B, params)
build_s = build_s or time.perf_counter() - t

# Search
for q8, qb, qt in zip(warm8, warmb, warm_texts):
    run_search(q8, qb, qt)
if params.get("cache") == "cold":
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
found = np.zeros((len(Q8), K), dtype=np.int64)
lat_ns = np.zeros(len(Q8), dtype=np.int64)
for i, (q8, qb, qt) in enumerate(zip(Q8, Qb, texts)):
    t = time.perf_counter_ns()
    found[i] = run_search(q8, qb, qt)
    lat_ns[i] = time.perf_counter_ns() - t

# Score
recall = np.mean([len(set(f[:10]) & set(g)) / 10 for f, g in zip(found, gt)])
passages = passage_texts(slice_name, found.ravel())
hits = [[has_answer(p, ans) for p in passages[i * K:(i + 1) * K]] for i, ans in enumerate(answers)]
accuracy = np.mean([any(h[:10]) for h in hits])
accuracy20 = f"{np.mean([any(h[:20]) for h in hits]):.4f}" if K >= 20 else ""
ms = lat_ns / 1e6
if hasattr(run_search, "sizes"):
    print(f"merged candidates per question: mean {np.mean(run_search.sizes):.0f}, max {max(run_search.sizes)}")
p50, p95, p99 = np.percentile(ms, [50, 95, 99])

# Save
tag = "_".join(f"{k}{v}" for k, v in params.items())
run = f"{datetime.now():%Y%m%d-%H%M%S}_{slice_name}_{setup}" + (f"_{tag}" if tag else "") + f"_{qset}"
Path("results/latencies").mkdir(parents=True, exist_ok=True)
np.save(f"results/latencies/{run}.npy", lat_ns)
row = {
    "run": run, "slice": slice_name, "setup": setup, "params": json.dumps(params),
    "qset": qset, "questions": len(Q8),
    "recall@10": f"{recall:.4f}", "accuracy@10": f"{accuracy:.4f}", "accuracy@20": accuracy20,
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
