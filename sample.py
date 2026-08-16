"""
Generate text from a trained checkpoint.

    python sample.py --prompt "ROMEO:" --n 400
"""
from __future__ import annotations

import argparse

import numpy as np

from gpt.data import encode
from gpt.model import GPT, softmax


def load(path):
    d = np.load(path, allow_pickle=False)
    vocab, block, nl, nh, ne = (int(v) for v in d["config"])
    model = GPT(vocab, block, nl, nh, ne)
    for k, arr in model.params().items():
        arr[...] = d[f"p/{k}"]
    itos = {i: str(c) for i, c in enumerate(d["chars"])}
    stoi = {c: i for i, c in itos.items()}
    return model, stoi, itos


def generate(model, stoi, itos, prompt, n, temperature=0.8, top_k=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = encode(prompt, stoi)[None, :]
    for _ in range(n):
        cond = idx[:, -model.block_size:]
        logits = model.forward(cond)[0, -1] / temperature
        if top_k:
            kth = np.sort(logits)[-top_k]
            logits = np.where(logits < kth, -np.inf, logits)
        p = softmax(logits)
        nxt = rng.choice(len(p), p=p)
        idx = np.concatenate([idx, [[nxt]]], axis=1)
    return "".join(itos[int(i)] for i in idx[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint.npz")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, stoi, itos = load(args.ckpt)
    print(generate(model, stoi, itos, args.prompt, args.n,
                   args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
