"""Adam optimizer (operating in place on the model's parameter arrays),
gradient-norm clipping, and the learning-rate schedule."""
from __future__ import annotations

import numpy as np


class Adam:
    """Adam with *decoupled* weight decay (i.e. AdamW).

    Decay is applied directly to the weights (p -= lr * wd * p) rather than
    folded into the gradient, so it isn't rescaled by the adaptive 1/sqrt(v)
    term — the Loshchilov & Hutter fix. Following GPT-2 practice, only 2-D+
    tensors (matmul weights, embeddings) are decayed; LayerNorm gains/biases
    and Linear biases are left alone, since shrinking those toward zero is
    just a regulariser on the wrong thing.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = params            # {name: array} — updated in place, by reference
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.decay = {k: v.ndim >= 2 for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            if self.wd and self.decay[k]:
                p *= 1.0 - self.lr * self.wd                    # decoupled decay
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


def lr_at(step, max_lr, total_steps, warmup=0, min_lr=0.0):
    """Linear warmup to max_lr over `warmup` steps, then cosine decay to min_lr
    at total_steps (the GPT-2 / nanoGPT schedule). Steps are 1-based."""
    if warmup and step <= warmup:
        return max_lr * step / warmup
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (1.0 + np.cos(np.pi * progress)) * (max_lr - min_lr)
