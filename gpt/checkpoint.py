"""
Save / load a model (+ optionally the optimizer) as a single .npz.

Layout inside the archive:
    chars        the vocabulary, index -> character
    config       [vocab_size, block_size, n_layer, n_head, n_embd]
    step         training step the checkpoint was written at
    p/<name>     one array per model parameter
    opt/m/<name>, opt/v/<name>, opt/t      Adam state (only if an optimizer is passed)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .model import GPT
from .optim import Adam


def save(path, model, itos, cfg, step=0, opt=None):
    arrays = {
        "chars": np.array([itos[i] for i in range(len(itos))]),
        "config": np.array(cfg, dtype=np.int64),
        "step": np.array(step, dtype=np.int64),
        **{f"p/{k}": v for k, v in model.params().items()},
    }
    if opt is not None:
        arrays.update({f"opt/m/{k}": v for k, v in opt.m.items()})
        arrays.update({f"opt/v/{k}": v for k, v in opt.v.items()})
        arrays["opt/t"] = np.array(opt.t, dtype=np.int64)
    # write to a temp file and rename so a Ctrl-C mid-save can't leave a
    # truncated checkpoint behind
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load(path, opt=None):
    """Rebuild the model from a checkpoint. Returns (model, stoi, itos, step).
    If an Adam instance bound to the model's params is passed, its moments and
    step counter are restored too (when the checkpoint has them)."""
    d = np.load(path, allow_pickle=False)
    vocab, block, nl, nh, ne = (int(v) for v in d["config"])
    dtype = d["p/wte.W"].dtype                        # float32 or float64, as trained
    model = GPT(vocab, block, nl, nh, ne, dtype=dtype).eval()  # inference mode; train.py re-enables dropout
    for k, arr in model.params().items():
        arr[...] = d[f"p/{k}"]
    itos = {i: str(c) for i, c in enumerate(d["chars"])}
    stoi = {c: i for i, c in itos.items()}
    step = int(d["step"]) if "step" in d.files else 0
    if opt is not None and "opt/t" in d.files:
        for k in opt.params:
            opt.m[k][...] = d[f"opt/m/{k}"]
            opt.v[k][...] = d[f"opt/v/{k}"]
        opt.t = int(d["opt/t"])
    return model, stoi, itos, step
