# /// script
# requires-python = ">=3.12"
# dependencies = ["cohere>=5", "faiss-cpu", "numpy", "usearch"]
# ///
# A small search page over the full corpus with the winning setup: the query is embedded
# with Cohere (same model as the passages), then binary HNSW (ef 250, depth 250) and
# IVF-PQ (nprobe 128, depth 200) run in parallel, their candidates are merged and
# rescored with exact int8 vectors read from the SSD, and the top 10 passages are shown.
# Each search makes one Cohere embed call (the trial key allows 2,000 embed inputs a minute).
# Usage: uv run --env-file .env scripts/serve.py   (then open http://127.0.0.1:8000)

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cohere
import faiss
import numpy as np
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


def answer(query):
    t0 = time.perf_counter()
    e = co.embed(model="embed-multilingual-v3.0", input_type="search_query",
                 texts=[query], embedding_types=["int8", "ubinary"]).embeddings
    t1 = time.perf_counter()
    ids = retrieve(np.array(e.int8[0], dtype=np.int8), np.array(e.ubinary[0], dtype=np.uint8))
    t2 = time.perf_counter()
    texts = [os.pread(text_fd, int(offsets[i + 1] - offsets[i]), int(offsets[i])).decode() for i in ids]
    return {"embed_ms": round((t1 - t0) * 1e3, 1), "search_ms": round((t2 - t1) * 1e3, 2),
            "passages": [{"id": int(i), "text": s} for i, s in zip(ids, texts)]}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wikipedia search</title>
<style>
  :root { --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --card: #fff; --line: #ddd; --accent: #2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #111; --fg: #e8e8e8; --muted: #999; --card: #1b1b1b; --line: #333; --accent: #6ea8fe; }
  }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 20px; font-size: 14px; }
  form { display: flex; gap: 8px; }
  input { flex: 1; padding: 10px 12px; font: inherit; border: 1px solid var(--line); border-radius: 6px;
          background: var(--card); color: var(--fg); }
  button { padding: 10px 16px; font: inherit; border: 0; border-radius: 6px; background: var(--accent);
           color: #fff; cursor: pointer; }
  #stats { color: var(--muted); font-size: 14px; margin: 16px 0 8px; min-height: 20px; }
  ol { padding-left: 24px; }
  li { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px;
       margin-bottom: 10px; }
  li small { color: var(--muted); display: block; }
</style></head>
<body><main>
  <h1>Wikipedia search</h1>
  <p class="sub">41.5M English Wikipedia passages. Binary HNSW + IVF-PQ union, int8 rescore from SSD.</p>
  <form id="f"><input id="q" placeholder="Ask a question" autofocus><button>Search</button></form>
  <div id="stats"></div>
  <ol id="results"></ol>
</main>
<script>
document.getElementById("f").onsubmit = async (ev) => {
  ev.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  const stats = document.getElementById("stats"), list = document.getElementById("results");
  stats.textContent = "Searching...";
  list.replaceChildren();
  const r = await fetch("/search?q=" + encodeURIComponent(q));
  if (!r.ok) { stats.textContent = "Error: " + await r.text(); return; }
  const d = await r.json();
  stats.textContent = `Embedding (Cohere API) ${d.embed_ms} ms · retrieval ${d.search_ms} ms`;
  for (const p of d.passages) {
    const li = document.createElement("li"), id = document.createElement("small");
    id.textContent = "passage " + p.id;
    li.append(id, p.text);
    list.append(li);
  }
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", PAGE.encode())
        elif url.path == "/search":
            q = parse_qs(url.query).get("q", [""])[0].strip()
            if not q:
                return self.reply(400, "text/plain", b"empty query")
            try:
                self.reply(200, "application/json", json.dumps(answer(q)).encode())
            except Exception as e:  # e.g. Cohere rate limit
                self.reply(502, "text/plain", str(e).encode())
        else:
            self.reply(404, "text/plain", b"not found")

    def reply(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)


# One request at a time, like the benchmark. Local only.
print("open http://127.0.0.1:8000", flush=True)
HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
