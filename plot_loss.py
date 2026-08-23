"""
Plot the train/val loss curve written by train.py (--log).

    python plot_loss.py                       # train_log.csv -> docs/loss.png
    python plot_loss.py run.csv -o run.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    cols["step"] = [int(v) for v in cols["step"]]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="train_log.csv")
    ap.add_argument("-o", "--out", default="docs/loss.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = read_log(args.log)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(d["step"], d["train_loss"], label="train", color="#3b82f6")
    ax.plot(d["step"], d["val_loss"], label="val", color="#f59e0b")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats / char)")
    ax.set_title("GPT from scratch — tiny-shakespeare")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}  (final val {d['val_loss'][-1]:.3f} at step {d['step'][-1]})")


if __name__ == "__main__":
    main()
