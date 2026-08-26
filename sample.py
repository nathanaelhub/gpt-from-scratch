"""
Generate text from a trained checkpoint.

    python sample.py --prompt "ROMEO:" --n 400
"""
from __future__ import annotations

import argparse

import numpy as np

from gpt import checkpoint
from gpt.data import encode
from gpt.model import softmax


def generate(model, stoi, itos, prompt, n, temperature=0.8, top_k=None, seed=0,
             use_cache=True):
    """Sample n characters after `prompt`.

    With use_cache each step feeds only the newest token through the model and
    attends to cached keys/values, instead of re-running the whole context.
    Learned absolute positions mean the cache can't simply slide, so once the
    context window is full it is rebuilt from the most recent block_size//2
    tokens (one short forward) and filling resumes. That keeps every step at
    single-token cost at the price of the model seeing between block_size/2
    and block_size tokens of context rather than always block_size.
    use_cache=False recomputes the full window every step (the exact original
    behaviour, and identical output while the prompt + n fit in one window).
    """
    if not prompt:
        raise ValueError("prompt must be at least one character")
    rng = np.random.default_rng(seed)
    idx = encode(prompt, stoi)[None, :]
    B = model.block_size
    cache = model.new_cache() if use_cache else None
    pending = idx[:, -B:]                      # tokens the cache hasn't seen yet
    for _ in range(n):
        if cache is None:
            logits = model.forward(idx[:, -B:])[0, -1]
        else:
            if model.cache_len(cache) + pending.shape[1] > B:
                cache = model.new_cache()      # window full: restart from recent half
                pending = idx[:, -(B // 2):]
            logits = model.forward(pending, cache)[0, -1]
        logits = logits / temperature
        if top_k:
            kth = np.sort(logits)[-top_k]
            logits = np.where(logits < kth, -np.inf, logits)
        p = softmax(logits)
        nxt = rng.choice(len(p), p=p)
        idx = np.concatenate([idx, [[nxt]]], axis=1)
        pending = np.array([[nxt]], dtype=np.int64)
    return "".join(itos[int(i)] for i in idx[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint.npz")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the full context every step instead of using the KV cache")
    args = ap.parse_args()

    model, stoi, itos, _ = checkpoint.load(args.ckpt)
    print(generate(model, stoi, itos, args.prompt, args.n,
                   args.temperature, args.top_k, args.seed, use_cache=not args.no_cache))


if __name__ == "__main__":
    main()
