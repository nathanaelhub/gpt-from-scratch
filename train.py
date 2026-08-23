"""
Train the from-scratch GPT on a character corpus.

    python train.py --steps 2000            # tiny-shakespeare, ~few minutes on CPU

Writes a checkpoint (params + vocab + config) that sample.py can generate from.
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from gpt import checkpoint
from gpt.data import CharData
from gpt.model import GPT
from gpt.optim import Adam, clip_grad_norm, lr_at


def estimate_loss(model, data, batch_size, iters=20):
    out = {}
    for split in ("train", "val"):
        out[split] = float(np.mean([model.loss(*data.get_batch(split, batch_size))
                                     for _ in range(iters)]))
    return out


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
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="decoupled (AdamW) weight decay on matmul/embedding weights")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip the global gradient norm to this value (0 = off)")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="evaluate and write a checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=0, help="model init + batch sampling seed")
    ap.add_argument("--out", default="checkpoint.npz")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from --out (params, Adam state, and step)")
    ap.add_argument("--log", default="train_log.csv",
                    help="CSV of step, train/val loss, lr, grad norm ('' to disable)")
    args = ap.parse_args()

    data = CharData(args.data, args.block_size, seed=args.seed)
    if args.resume:
        model, _, _, start = checkpoint.load(args.out)
        opt = Adam(model.params(), lr=args.lr, weight_decay=args.weight_decay)
        checkpoint.load(args.out, opt=opt)
        print(f"resumed {args.out} at step {start}")
    else:
        model = GPT(data.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd,
                    seed=args.seed)
        opt = Adam(model.params(), lr=args.lr, weight_decay=args.weight_decay)
        start = 0
    cfg = [model.wte.W.shape[0], model.block_size, len(model.blocks),
           model.blocks[0].attn.n_head, model.wte.W.shape[1]]
    n_params = sum(p.size for p in model.params().values())
    print(f"corpus vocab {data.vocab_size} | {n_params:,} parameters | {args.steps} steps")

    def save(step):
        checkpoint.save(args.out, model, data.itos, cfg, step=step, opt=opt)

    log = None
    if args.log:
        # append when resuming so the curve stays continuous; otherwise start fresh
        f = open(args.log, "a" if args.resume else "w", newline="", encoding="utf-8")
        log = csv.writer(f)
        if not args.resume:
            log.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "seconds"])

    t0 = time.time()
    step = start
    try:
        for step in range(start + 1, args.steps + 1):
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
                if log:
                    log.writerow([step, f"{e['train']:.4f}", f"{e['val']:.4f}",
                                  f"{opt.lr:.3e}", f"{gnorm:.3f}", f"{time.time() - t0:.1f}"])
                    f.flush()
                save(step)
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}; saving")
    save(step)
    if log:
        f.close()
    print(f"saved {args.out} (step {step})")


if __name__ == "__main__":
    main()
