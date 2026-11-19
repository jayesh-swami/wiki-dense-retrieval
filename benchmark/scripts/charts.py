# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
# Charts for benchmark/README.md, from benchmark/results/results.csv:
#   charts/full_recall_p99.png   41.5M setups on tune, cold cache: recall@10 against p99
#   charts/final_check.png       winner vs runner-up on val, 5 runs each: p50 / p95 / p99
# Usage: charts.py   (from the repo root)

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("benchmark/charts")
OUT.mkdir(exist_ok=True)
rows = [r for r in csv.DictReader(open("benchmark/results/results.csv")) if r["slice"] == "full"]
for r in rows:
    r["p"] = json.loads(r["params"])

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1. Every 41.5M setup on tune with a cold cache (last run of each setting; fadvise on one
#    thread, except the 250/250 union, which ran after the 4-thread change).
def setting(r):
    p = r["p"]
    if r["setup"] == "hnsw_binary":
        return f"HNSW M{p['M']} {p['ef']}/{p['depth']}"
    if r["setup"] == "ivf":
        return f"IVF-PQ {p['nprobe']}/{p['depth']}"
    return f"{p['h.ef']}/{p['h.depth']} + {p['i.nprobe']}/{p['i.depth']}"


tune = {}
for r in rows:
    if r["qset"] == "tune" and r["p"].get("cache") == "cold" and "readers" not in r["p"]:
        key = (r["setup"], setting(r))
        if "k" not in r["p"] or key not in tune:  # later runs replace earlier ones, k=20 runs don't
            tune[key] = r

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
style(ax)
families = [("hnsw_binary", "Binary HNSW + int8 rescore", BLUE, "o"),
            ("ivf", "IVF-PQ + int8 rescore", ORANGE, "s"),
            ("union", "Union of both + int8 rescore", AQUA, "D")]
for setup, label, color, marker in families:
    pts = [(float(r["p99_ms"]), float(r["recall@10"]), name) for (s, name), r in tune.items() if s == setup]
    ax.scatter([x for x, _, _ in pts], [y for _, y, _ in pts], s=70, color=color, marker=marker, label=label,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    labels = {"250/250 + 128/200": ("winner: union 250/250 + 128/200", (-10, 6), "right"),
              "HNSW M32 600/300": ("HNSW M32 600/300", (8, -14), "left"),
              "HNSW M16 1000/1000": ("HNSW M16 1000/1000", (-8, 7), "right"),
              "IVF-PQ 384/300": ("IVF-PQ 384/300", (8, -4), "left")}
    for x, y, name in pts:
        if name in labels:
            text, offset, ha = labels[name]
            ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=9.5, color=INK, ha=ha)
ax.axhline(0.95, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
ax.text(0.15, 0.9485, "recall bar 0.95", ha="left", va="top", fontsize=9.5, color=MUTED)
ax.set_xlim(0, 10)
ax.set_xlabel("p99 latency (ms), one query at a time, cold cache")
ax.set_ylabel("recall@10 against exact search")
ax.set_title("41.5M passages, tune questions: every setup tested", loc="left", fontsize=13, pad=12)
ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "full_recall_p99.png")

# 2. Final check: 5 alternating val runs each of the two best unions (4-thread fadvise).
runs = {"250/250": [], "300/300": []}
for r in rows:
    p = r["p"]
    if (r["qset"] == "val" and r["setup"] == "union" and p.get("cache") == "cold" and p["i.nprobe"] == "128"
            and p["i.depth"] == "200" and p["h.ef"] in ("250", "300") and "135500" <= r["run"][9:15] <= "140100"):
        runs[f"{p['h.ef']}/{p['h.depth']}"].append([float(r[k]) for k in ("p50_ms", "p95_ms", "p99_ms")])
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
style(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.8)
ax.grid(False, axis="x")
x = np.arange(3)
for i, (name, color, label) in enumerate([("250/250", BLUE, "Winner: HNSW 250/250 + IVF-PQ 128/200"),
                                           ("300/300", ORANGE, "Runner-up: HNSW 300/300 + IVF-PQ 128/200")]):
    v = np.array(runs[name])
    med = np.median(v, axis=0)
    xs = x + (i - 0.5) * 0.36
    ax.bar(xs, med, width=0.34, color=color, label=f"{label} (median of {len(v)} runs)", zorder=2)
    for j in range(3):
        ax.scatter(np.full(len(v), xs[j]), v[:, j], s=22, color=SURFACE, edgecolors=INK, linewidths=1, zorder=3)
        ax.text(xs[j], med[j] / 2, f"{med[j]:.2f}", ha="center", va="center", fontsize=10, color="#ffffff")
ax.set_xticks(x, ["p50", "p95", "p99"])
ax.set_ylabel("latency (ms)")
ax.set_ylim(0, 7)
ax.set_title("Final check on val, 41.5M passages, cold cache: bars are medians, dots are single runs",
             loc="left", fontsize=12, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "final_check.png")
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
