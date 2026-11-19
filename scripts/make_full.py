# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyarrow"]
# ///
# Write the full English corpus (all 41.5M passages) in the files search reads
# at random, streaming the raw parquet files in order. Row number = passage id.
#   int8.npy     (N, 1024) int8. Header padded to 4096 bytes, so every vector
#                sits inside one 4 KB SSD page.
#   binary.npy   (N, 128) uint8
#   text.bin     all passage texts back to back (UTF-8)
#   offsets.npy  (N + 1) int64: passage i is text.bin[offsets[i]:offsets[i+1]]
# Writes data/slices/full/. Usage: make_full.py

import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

files = sorted(glob.glob("data/raw/en/*.parquet"))
out = Path("data/slices/full")
out.mkdir(parents=True, exist_ok=True)
n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def open_aligned(path, shape, dtype):
    # A .npy file whose header is padded to 4096 bytes; np.load reads it as usual.
    header = repr({"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                   "fortran_order": False, "shape": shape}).encode()
    header = header + b" " * (4096 - 10 - len(header) - 1) + b"\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header)
    return np.memmap(path, dtype=dtype, mode="r+", offset=4096, shape=shape)


int8 = open_aligned(out / "int8.npy", (n, 1024), np.int8)
binary = np.lib.format.open_memmap(out / "binary.npy", mode="w+", dtype=np.uint8, shape=(n, 128))
offsets = np.lib.format.open_memmap(out / "offsets.npy", mode="w+", dtype=np.int64, shape=(n + 1,))
offsets[0] = 0

i = 0
with open(out / "text.bin", "wb") as text:
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["text", "emb_int8", "emb_ubinary"]):
            k = batch.num_rows
            int8[i:i + k] = pc.list_flatten(batch["emb_int8"]).to_numpy().reshape(k, 1024)
            binary[i:i + k] = pc.list_flatten(batch["emb_ubinary"]).to_numpy().reshape(k, 128)
            # copy the text bytes straight from Arrow's buffers (no per-row Python)
            t = batch["text"].cast(pa.large_string())
            o = np.frombuffer(t.buffers()[1], dtype=np.int64)[t.offset:t.offset + k + 1]
            text.write(memoryview(t.buffers()[2])[o[0]:o[-1]])
            offsets[i + 1:i + k + 1] = offsets[i] + (o[1:] - o[0])
            i += k
        print(f, i)

assert i == n
int8.flush(), binary.flush(), offsets.flush()
print("full", n, "passages")
