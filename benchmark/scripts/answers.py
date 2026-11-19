# Answer matching shared by the scripts: normalise like SQuAD/DPR, then look for
# each answer as whole words (space-padded substring) in the passage.

import os
import re
import string
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_answers(qset):
    return pq.read_table(f"data/questions/{qset}/questions.parquet")["answer"].to_pylist()


def passage_texts(slice_name, ids):
    # Texts for passage ids: from text.bin + offsets.npy (full corpus), else passages.parquet.
    d = Path("data/slices") / slice_name
    if not (d / "text.bin").exists():
        return pq.read_table(d / "passages.parquet", columns=["text"])["text"].take(ids).to_pylist()
    offsets = np.load(d / "offsets.npy", mmap_mode="r")
    fd = os.open(d / "text.bin", os.O_RDONLY)
    texts = [os.pread(fd, int(offsets[i + 1] - offsets[i]), int(offsets[i])).decode() for i in ids]
    os.close(fd)
    return texts


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def has_answer(passage, answers):
    p = f" {normalize(passage)} "
    return any(f" {normalize(a)} " in p for a in answers)
