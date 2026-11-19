# /// script
# requires-python = ">=3.12"
# dependencies = ["cohere>=5", "faiss-cpu", "fastapi", "numpy", "usearch", "uvicorn"]
# ///
# Search API over the full corpus with the winning setup, plus the web page (web/dist).
# A query is embedded with Cohere (same model as the passages), then binary HNSW (ef 250,
# depth 250) and IVF-PQ (nprobe 128, depth 200) run in parallel, their candidates are
# merged and rescored with exact int8 vectors read from the SSD, and the top 10 are returned.
# Each search makes one Cohere embed call (the trial key allows 100 calls a minute).
# GET /api/warm?seconds=N keeps the CPU caches warm for N seconds by running stored val
# question vectors through retrieval back to back (no Cohere calls); seconds=0 stops it.
# Loads the indexes at startup (~20-30 s) and holds ~18 GiB of RAM (14.8 GB of index files).
# Usage: uv run --env-file .env api/main.py   (from the repo root; then open http://127.0.0.1:8000)

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cohere
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from usearch.index import Index, MetricKind, search

D = "data/slices/full"
faiss.omp_set_num_threads(1)
co = cohere.ClientV2(api_key=os.environ["COHERE_KEY"])

t = time.perf_counter()
X = np.load(f"{D}/int8.npy", mmap_mode="r")
offsets = np.load(f"{D}/offsets.npy", mmap_mode="r")
fd, text_fd = os.open(f"{D}/int8.npy", os.O_RDONLY), os.open(f"{D}/text.bin", os.O_RDONLY)
hnsw = Index(ndim=1024, metric="hamming", dtype="b1", connectivity=16, expansion_add=128)
hnsw.load("data/indexes/full/hnsw_b1_M16_efc128.usearch")
hnsw.expansion_search = 250
ivf = faiss.read_index("data/indexes/full/IVF65536_PQ128x4fs.faiss")
centres = faiss.IndexHNSWFlat(1024, 32, faiss.METRIC_INNER_PRODUCT)
centres.add(ivf.quantizer.reconstruct_n(0, ivf.nlist))
centres.hnsw.efSearch = 256
ivf.quantizer, ivf.own_fields, ivf.referenced_objects = centres, False, [centres]
ivf.nprobe = 128
pool, advisers = ThreadPoolExecutor(1), ThreadPoolExecutor(4)
one_at_a_time = threading.Lock()  # searches run one at a time, as in the benchmark
print(f"loaded in {time.perf_counter() - t:.0f} s", flush=True)


def advise(offs):
    for o in offs:
        os.posix_fadvise(fd, o, 1024, os.POSIX_FADV_WILLNEED)


def retrieve(q8, qb):
    f = pool.submit(lambda: ivf.search(q8[None].astype(np.float32), 200)[1][0])
    cand = np.union1d(hnsw.search(qb, 250, threads=1).keys.astype(np.int64), f.result())
    offs = (cand * 1024 + X.offset).tolist()
    list(advisers.map(advise, [offs[k::4] for k in range(4)]))
    return cand[search(np.array(X[cand]), q8, 10, MetricKind.IP, exact=True, threads=1).keys]


Q8, QB = np.load("data/questions/val/q_int8.npy"), np.load("data/questions/val/q_binary.npy")
warm_until, warmer = 0.0, None


def keep_warm():
    i = 0
    while time.perf_counter() < warm_until:
        with one_at_a_time:
            retrieve(Q8[i % len(Q8)], QB[i % len(QB)])
        i += 1
        time.sleep(0.0005)  # let a waiting page search take the lock


app = FastAPI()


@app.get("/api/warm")
def warm(seconds: float = 600):  # seconds=0 stops the warm-up
    global warm_until, warmer
    warm_until = time.perf_counter() + seconds
    if seconds > 0 and not (warmer and warmer.is_alive()):
        warmer = threading.Thread(target=keep_warm, daemon=True)
        warmer.start()
    return {"warming_s": seconds}


@app.get("/api/search")
def search_passages(q: str):
    q = q.strip()
    if not q:
        raise HTTPException(400, "empty query")
    t0 = time.perf_counter()
    try:
        e = co.embed(model="embed-multilingual-v3.0", input_type="search_query",
                     texts=[q], embedding_types=["int8", "ubinary"]).embeddings
    except Exception as err:  # e.g. Cohere rate limit
        raise HTTPException(502, f"Cohere embed failed: {err}")
    t1 = time.perf_counter()
    with one_at_a_time:
        t2 = time.perf_counter()
        ids = retrieve(np.array(e.int8[0], dtype=np.int8), np.array(e.ubinary[0], dtype=np.uint8))
        t3 = time.perf_counter()
    texts = [os.pread(text_fd, int(offsets[i + 1] - offsets[i]), int(offsets[i])).decode() for i in ids]
    return {"embed_ms": round((t1 - t0) * 1e3, 1), "search_ms": round((t3 - t2) * 1e3, 2),
            "passages": [{"id": int(i), "text": s} for i, s in zip(ids, texts)]}


if Path("web/dist").exists():  # the built React app (make web)
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
