"""Adam optimizer (operating in place on the model's parameter arrays) and
gradient-norm clipping."""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = params            # {name: array} — updated in place, by reference
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, p in self.params.items():
            g = grads[k]
            if self.wd:
                g = g + self.wd * p
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)   # in-place


def clip_grad_norm(grads, max_norm):
    """Scale all gradients in place so their global L2 norm is at most max_norm.

    Returns the norm *before* clipping (useful to log — a spike is the usual
    sign that the learning rate is too high). Clipping the global norm rather
    than each tensor keeps the update direction unchanged.
    """
    total = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= scale
    return total
