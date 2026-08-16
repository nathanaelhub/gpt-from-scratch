"""Character-level dataset: build a vocab from a text file and serve batches."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class CharData:
    def __init__(self, path, block_size, split=0.9, seed=0):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n = int(len(data) * split)
        self.train, self.val = data[:n], data[n:]
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def get_batch(self, split, batch_size):
        d = self.train if split == "train" else self.val
        ix = self.rng.integers(0, len(d) - self.block_size - 1, batch_size)
        x = np.stack([d[i:i + self.block_size] for i in ix])
        y = np.stack([d[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

    def encode(self, s):
        return encode(s, self.stoi)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def encode(s, stoi):
    """Map a string to token ids, failing loudly on characters outside the vocab
    (silently substituting a token would corrupt the prompt without warning)."""
    unknown = sorted(set(s) - stoi.keys())
    if unknown:
        raise ValueError(f"characters not in the model's vocabulary: {unknown!r}")
    return np.array([stoi[c] for c in s], dtype=np.int64)
