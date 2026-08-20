"""
Train the from-scratch GPT on a character corpus.

    python train.py --steps 2000            # tiny-shakespeare, ~few minutes on CPU

Writes a checkpoint (params + vocab + config) that sample.py can generate from.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gpt.data import CharData
from gpt.model import GPT
from gpt.optim import Adam, clip_grad_norm, lr_at


def estimate_loss(model, data, batch_size, iters=20):
    out = {}
    for split in ("train", "val"):
        out[split] = float(np.mean([model.loss(*data.get_batch(split, batch_size))
                                     for _ in range(iters)]))
    return out


def save(path, model, data, cfg):
    np.savez(path,
             chars=np.array([data.itos[i] for i in range(data.vocab_size)]),
             config=np.array(cfg),
             **{f"p/{k}": v for k, v in model.params().items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    ap.add_argument("--warmup", type=int, default=100,
                    help="linear warmup steps before cosine decay (0 = constant lr)")
    ap.add_argument("--min-lr", type=float, default=3e-4,
                    help="learning rate the cosine schedule decays to by the last step")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip the global gradient norm to this value (0 = off)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default="checkpoint.npz")
    args = ap.parse_args()

    data = CharData(args.data, args.block_size)
    model = GPT(data.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd)
    opt = Adam(model.params(), lr=args.lr)
    n_params = sum(p.size for p in model.params().values())
    print(f"corpus vocab {data.vocab_size} | {n_params:,} parameters | {args.steps} steps")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = data.get_batch("train", args.batch_size)
        opt.lr = lr_at(step, args.lr, args.steps, args.warmup, args.min_lr)
        model.loss(x, y)
        grads = model.backward()
        gnorm = clip_grad_norm(grads, args.grad_clip)
        opt.step(grads)
        if step == 1 or step % args.eval_every == 0:
            e = estimate_loss(model, data, args.batch_size)
            print(f"  step {step:5d} | train {e['train']:.3f} | val {e['val']:.3f} "
                  f"| lr {opt.lr:.2e} | grad norm {gnorm:.2f} | {time.time() - t0:.0f}s")

    save(args.out, model, data,
         [data.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
