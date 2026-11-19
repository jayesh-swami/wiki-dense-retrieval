# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu", "numpy"]
# ///
# Build IVF<nlist> + PQ128x4fs (fast-scan, 64 B per passage) with inner product for a
# slice, without holding all int8 vectors as float: train on a sample (FAISS needs
# float32), then add in batches of 1M. Passages are assigned to cells through an HNSW
# graph over the centres (IVF<nlist>_HNSW32), so adding 41.5M takes minutes, not hours.
# Writes data/indexes/<slice>/IVF<nlist>_PQ128x4fs.faiss (+ .build_s), where bench.py
# looks for it (run with lists=PQ128x4fs nlist=<nlist> coarse=hnsw).
# Usage: build_ivfpq.py full 65536 1300000

import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

name, nlist, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
X = np.load(f"data/slices/{name}/int8.npy", mmap_mode="r")
path = Path(f"data/indexes/{name}/IVF{nlist}_PQ128x4fs.faiss")
faiss.omp_set_num_threads(os.cpu_count())

index = faiss.index_factory(1024, f"IVF{nlist}_HNSW32,PQ128x4fs", faiss.METRIC_INNER_PRODUCT)
t = time.perf_counter()
sample = np.sort(np.random.default_rng(0).choice(len(X), n, replace=False))
index.train(X[sample].astype(np.float32))
del sample
print(f"trained in {time.perf_counter() - t:.0f} s", flush=True)

faiss.downcast_index(index.quantizer).hnsw.efSearch = 64  # cell assignment quality
for i in range(0, len(X), 1_000_000):
    index.add(X[i:i + 1_000_000].astype(np.float32))
    print(f"added {i + 1_000_000:,}", flush=True)

path.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(path))
path.with_suffix(".build_s").write_text(f"{time.perf_counter() - t:.1f}")
sizes = np.array([index.invlists.list_size(c) for c in range(nlist)])
print(f"built in {time.perf_counter() - t:.0f} s; list sizes mean {sizes.mean():.0f}, "
      f"p99 {np.percentile(sizes, 99):.0f}, max {sizes.max()}")
