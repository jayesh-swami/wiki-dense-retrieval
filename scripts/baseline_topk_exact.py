# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
# Exact (brute-force) top K passages (default 100) for every question, by int8 dot product.
# This is the ground truth every index is compared against.
# Reads data/slices/<name>/int8.npy and data/questions/<set>/q_int8.npy
# Writes baselines/<name>/top-<K>/gt_<set>.npy: (questions, K) passage row ids, best first
# Usage: baseline_topk_exact.py 5m [K]

import sys
import time
from pathlib import Path

import numpy as np

CHUNK = 100_000  # passages scored at once; keeps the score matrix around 1.4 GB

name = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
d = Path("data/slices") / name
X = np.load(d / "int8.npy", mmap_mode="r")
out = Path("baselines") / name / f"top-{K}"
out.mkdir(parents=True, exist_ok=True)

for qset in ["val", "tune"]:
    Q = np.load(f"data/questions/{qset}/q_int8.npy").astype(np.float32)
    best_s = np.full((len(Q), K), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(Q), K), dtype=np.int64)
    t = time.time()

    for start in range(0, len(X), CHUNK):
        # float32 holds these int8 dot products exactly (max about 16.5M < 2^24)
        s = Q @ X[start:start + CHUNK].astype(np.float32).T
        top = np.argpartition(s, -K, axis=1)[:, -K:]
        cand_s = np.concatenate([best_s, np.take_along_axis(s, top, 1)], axis=1)
        cand_i = np.concatenate([best_i, top + start], axis=1)
        keep = np.argpartition(cand_s, -K, axis=1)[:, -K:]
        best_s = np.take_along_axis(cand_s, keep, 1)
        best_i = np.take_along_axis(cand_i, keep, 1)

    order = np.argsort(-best_s, axis=1)
    np.save(out / f"gt_{qset}.npy", np.take_along_axis(best_i, order, 1).astype(np.int32))
    print(f"{qset}: {len(Q)} questions, {time.time() - t:.0f}s")
