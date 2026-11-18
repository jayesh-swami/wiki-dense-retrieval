# Answer matching shared by the scripts: normalise like SQuAD/DPR, then look for
# each answer as whole words (space-padded substring) in the passage.

import re
import string


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def has_answer(passage, answers):
    p = f" {normalize(passage)} "
    return any(f" {normalize(a)} " in p for a in answers)
