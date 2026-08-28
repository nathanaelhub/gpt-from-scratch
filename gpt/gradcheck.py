"""
Numerically verify every hand-derived gradient in model.py.

For a tiny GPT on random data, compare each parameter's analytic gradient
(from the backward pass) to the central finite difference
(L(w+eps) - L(w-eps)) / (2 eps) at a handful of random entries. If the chain
rule is coded correctly, the max relative error is ~1e-6.
"""
from __future__ import annotations

import numpy as np

from .model import GPT


def gradcheck(eps=1e-5, n_per_param=6, seed=0, verbose=True, dropout=0.0):
    model = GPT(vocab_size=13, block_size=8, n_layer=2, n_head=2, n_embd=16, seed=seed,
                dropout=dropout)
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, 13, (2, 8))
    targets = rng.integers(0, 13, (2, 8))

    # with dropout on, every loss() must see the *same* masks, so restart the
    # mask stream before each call
    def loss():
        model.reseed_dropout(seed + 2)
        return model.loss(idx, targets)

    loss()
    grads = model.backward()
    params = model.params()

    worst_overall = 0.0
    for name, P in params.items():
        flat = P.ravel()
        gflat = np.asarray(grads[name]).ravel()
        picks = rng.choice(flat.size, min(n_per_param, flat.size), replace=False)
        worst = 0.0
        for i in picks:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss()
            flat[i] = orig - eps
            lm = loss()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-12)
            worst = max(worst, rel)
        worst_overall = max(worst_overall, worst)
        if verbose:
            flag = "ok" if worst < 1e-4 else "FAIL"
            print(f"  {name:22s} max rel err {worst:.2e}  [{flag}]")
    return worst_overall


if __name__ == "__main__":
    print("Gradient check (analytic vs. central differences):")
    worst = gradcheck()
    print(f"\nworst relative error across all parameters: {worst:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL")
