# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow"]
# ///
# Answer accuracy@N of the exact top-K: the share of val questions where one of
# the first N passages contains an answer string (normalised, whole words).
# Reads baselines/<name>/top-<K>/gt_val.npy, the slice text and the val answers.
# Adds or updates rows in baselines/accuracy.csv (slice, topk, at, accuracy)
# Usage: baseline_accuracy.py 5m 100 10,100

import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from answers import has_answer

name, K, ats = sys.argv[1], int(sys.argv[2]), [int(a) for a in sys.argv[3].split(",")]
assert max(ats) <= K, f"accuracy@{max(ats)} needs top-K with K >= {max(ats)}"

out = Path("baselines") / name / f"top-{K}"
gt = np.load(out / "gt_val.npy")[:, :max(ats)]
questions = pq.read_table("data/questions/val/questions.parquet")
answers = questions["answers" if "answers" in questions.column_names else "answer"].to_pylist()
text = pq.read_table(f"data/slices/{name}/passages.parquet", columns=["text"])["text"]

passages = text.take(gt.ravel()).to_pylist()
hits = np.zeros(gt.shape, dtype=bool)
for q, ans in enumerate(answers):
    for r in range(gt.shape[1]):
        hits[q, r] = has_answer(passages[q * gt.shape[1] + r], ans)

table = Path("baselines/accuracy.csv")
rows = list(csv.DictReader(table.open())) if table.exists() else []
rows = [r for r in rows if not (r["slice"] == name and r["topk"] == str(K) and int(r["at"]) in ats)]
for n in ats:
    acc = hits[:, :n].any(axis=1).mean()
    rows.append({"slice": name, "topk": str(K), "at": str(n), "accuracy": f"{acc:.4f}"})
    print(f"{name} top-{K} accuracy@{n}: {acc:.4f}")

rows.sort(key=lambda r: (r["slice"], int(r["topk"]), int(r["at"])))
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["slice", "topk", "at", "accuracy"])
    w.writeheader()
    w.writerows(rows)
