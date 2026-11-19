# /// script
# requires-python = ">=3.12"
# dependencies = ["cohere>=5", "numpy", "pyarrow"]
# ///
# Embed NQ-open questions with Cohere (same model as the Wikipedia passages).
# val  = all 3,610 validation questions (for reported numbers)
# tune = 3,000 random train questions (for picking index settings)
# Writes data/questions/<set>/{questions.parquet, q_int8.npy, q_binary.npy}

import os
import time
from pathlib import Path

import cohere
import numpy as np
import pyarrow.parquet as pq

NQ = Path("data/nq_open/nq_open")
OUT = Path("data/questions")

co = cohere.ClientV2(api_key=os.environ["COHERE_KEY"])

val = pq.read_table(NQ / "validation-00000-of-00001.parquet")
train = pq.read_table(NQ / "train-00000-of-00001.parquet")
picked = np.random.default_rng(42).choice(train.num_rows, 3000, replace=False)
tune = train.take(np.sort(picked))

for name, table in [("val", val), ("tune", tune)]:
    questions = table["question"].to_pylist()
    int8, binary = [], []
    for i in range(0, len(questions), 96):
        res = co.embed(
            model="embed-multilingual-v3.0",
            input_type="search_query",
            texts=questions[i:i + 96],
            embedding_types=["int8", "ubinary"],
        )
        int8 += res.embeddings.int8
        binary += res.embeddings.ubinary
        print(name, i + 96)
        time.sleep(3)  # trial limit is 2,000 inputs per minute

    (OUT / name).mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT / name / "questions.parquet")
    np.save(OUT / name / "q_int8.npy", np.array(int8, dtype=np.int8))
    np.save(OUT / name / "q_binary.npy", np.array(binary, dtype=np.uint8))
