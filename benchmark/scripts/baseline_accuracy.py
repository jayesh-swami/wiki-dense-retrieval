# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow"]
# ///
# Answer accuracy@N of the exact top-K: the share of val questions where one of
# the first N passages contains an answer string (normalised, whole words).
# Reads benchmark/baselines/<name>/top-<K>/gt_val.npy, the slice text and the val answers.
# Adds or updates rows in benchmark/baselines/accuracy.csv (slice, topk, at, accuracy)
# Usage: baseline_accuracy.py 5m 100 10,100

import csv
import sys
from pathlib import Path

import numpy as np

from answers import has_answer, load_answers, passage_texts

name, K, ats = sys.argv[1], int(sys.argv[2]), [int(a) for a in sys.argv[3].split(",")]
assert max(ats) <= K, f"accuracy@{max(ats)} needs top-K with K >= {max(ats)}"

out = Path("benchmark/baselines") / name / f"top-{K}"
gt = np.load(out / "gt_val.npy")[:, :max(ats)]
answers = load_answers("val")

passages = passage_texts(name, gt.ravel())
n = gt.shape[1]
hits = np.array([[has_answer(passages[q * n + r], ans) for r in range(n)] for q, ans in enumerate(answers)])

table = Path("benchmark/baselines/accuracy.csv")
rows = list(csv.DictReader(table.open())) if table.exists() else []
rows = [r for r in rows if not (r["slice"] == name and r["topk"] == str(K) and int(r["at"]) in ats)]
for at in ats:
    acc = hits[:, :at].any(axis=1).mean()
    rows.append({"slice": name, "topk": str(K), "at": str(at), "accuracy": f"{acc:.4f}"})
    print(f"{name} top-{K} accuracy@{at}: {acc:.4f}")

rows.sort(key=lambda r: (r["slice"], int(r["topk"]), int(r["at"])))
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["slice", "topk", "at", "accuracy"])
    w.writeheader()
    w.writerows(rows)
