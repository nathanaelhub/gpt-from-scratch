"""
Score a checkpoint on the *entire* validation split.

    python eval.py                        # checkpoint.npz on data/tinyshakespeare.txt
    python eval.py --ckpt run.npz --split train

train.py's running estimate averages 20 random batches; this sweeps every
non-overlapping block of the split once and reports the exact mean
cross-entropy, plus the two numbers people actually compare:

    perplexity      = exp(loss)     (effective vocabulary size per prediction)
    bits per char   = loss / ln 2   (compression-ratio view: ASCII is 8)
"""
from __future__ import annotations

import argparse

import numpy as np

from gpt import checkpoint
from gpt.data import CharData


def evaluate(model, ids, block_size, batch_size=64):
    """Mean cross-entropy over every token in `ids` (except the first, which
    has no context), predicting each from the preceding block."""
    n_blocks = (len(ids) - 1) // block_size
    if n_blocks == 0:
        raise ValueError(f"split has {len(ids)} tokens; need > block_size={block_size}")
    total, count = 0.0, 0
    for start in range(0, n_blocks, batch_size):
        rows = range(start, min(start + batch_size, n_blocks))
        x = np.stack([ids[i * block_size:(i + 1) * block_size] for i in rows])
        y = np.stack([ids[i * block_size + 1:(i + 1) * block_size + 1] for i in rows])
        total += model.loss(x, y) * x.size      # loss is a per-token mean
        count += x.size
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoint.npz")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model, stoi, itos, step = checkpoint.load(args.ckpt)
    data = CharData(args.data, model.block_size)
    if data.stoi != stoi:
        raise SystemExit("checkpoint vocabulary does not match --data")
    ids = data.val if args.split == "val" else data.train
    loss = evaluate(model, ids, model.block_size, args.batch_size)
    print(f"{args.ckpt} (step {step}) on {args.split}: {len(ids):,} chars")
    print(f"  cross-entropy  {loss:.4f} nats/char")
    print(f"  perplexity     {np.exp(loss):.2f}")
    print(f"  bits per char  {loss / np.log(2):.3f}")
    print(f"  (random baseline: ln({data.vocab_size}) = {np.log(data.vocab_size):.3f} nats)")


if __name__ == "__main__":
    main()
