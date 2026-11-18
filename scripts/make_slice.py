# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb", "numpy", "pyarrow"]
# ///
# Make a slice of whole Wikipedia articles, sampled across all 415 files.
# An article is kept when article_id % 1000 < K, so K=24 is about 1M passages
# and K=120 about 5M, and every smaller slice sits inside the bigger ones.
# Row number = passage id in every file below.
# Writes data/slices/<name>/{passages.parquet, int8.npy, binary.npy}
# Usage: make_slice.py 120 5m

import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

k, name = int(sys.argv[1]), sys.argv[2]
out = Path("data/slices") / name
out.mkdir(parents=True, exist_ok=True)

duckdb.sql("SET preserve_insertion_order = false; SET memory_limit = '6GB'; SET threads = 8")
duckdb.sql(f"""
    COPY (
        SELECT _id, title, text, emb_int8, emb_ubinary
        FROM 'data/raw/en/*.parquet'
        WHERE CAST(split_part(_id, '_', 2) AS BIGINT) % 1000 < {k}
    ) TO '{out}/passages.parquet'
""")

# Write the vectors batch by batch so the full array is never in RAM twice.
pf = pq.ParquetFile(out / "passages.parquet")
n = pf.metadata.num_rows
for col, dim, dtype, file in [("emb_int8", 1024, np.int8, "int8.npy"),
                              ("emb_ubinary", 128, np.uint8, "binary.npy")]:
    arr = np.lib.format.open_memmap(out / file, mode="w+", dtype=dtype, shape=(n, dim))
    i = 0
    for batch in pf.iter_batches(columns=[col], batch_size=100_000):
        rows = pc.list_flatten(batch[col]).to_numpy().reshape(-1, dim)
        arr[i:i + len(rows)] = rows
        i += len(rows)
    arr.flush()

print(name, n, "passages")
