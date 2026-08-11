"""
A GPT, from scratch, in NumPy — every gradient derived and coded by hand.

The architecture is the standard decoder-only transformer (as in GPT-2):

    idx ─► token emb + positional emb ─► [ Block ] x N ─► final LayerNorm ─► head ─► logits

Each Block is pre-norm with residual connections:

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Every layer is a small object with `forward` (which caches what the backward
pass needs) and `backward` (which returns the gradient w.r.t. its input and
stashes the gradients of its own parameters in `self.grads`). Nothing here calls
an autograd library — the chain rule is spelled out. `gradcheck.py` verifies the
whole thing against central finite differences.

Arrays are float64 so the gradient check is exact to ~1e-7; the model is tiny so
the speed cost doesn't matter.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- ops
def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# tanh-approximation GELU (what GPT-2 uses) and its exact derivative
_GC = np.sqrt(2.0 / np.pi)


def gelu(x):
    inner = _GC * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def dgelu(x):
    inner = _GC * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    dinner = _GC * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner


# ----------------------------------------------------------------------- layers
class Linear:
    """y = x @ W + b, over the last axis. x: (..., in), y: (..., out)."""

    def __init__(self, n_in, n_out, rng, scale=None):
        s = scale if scale is not None else np.sqrt(2.0 / n_in)
        self.W = (rng.standard_normal((n_in, n_out)) * s)
        self.b = np.zeros(n_out)
        self.grads = {}

    def params(self):
        return {"W": self.W, "b": self.b}

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # collapse every leading axis into one batch dimension for the matmuls
        xf = self.x.reshape(-1, self.W.shape[0])
        df = dout.reshape(-1, self.W.shape[1])
        self.grads["W"] = xf.T @ df
        self.grads["b"] = df.sum(0)
        return (dout @ self.W.T).reshape(self.x.shape)


class Embedding:
    """Row lookup: idx (int array) -> vectors. Gradient scatters back to rows."""

    def __init__(self, n, dim, rng):
        self.W = rng.standard_normal((n, dim)) * 0.02
        self.grads = {}

    def params(self):
        return {"W": self.W}

    def forward(self, idx):
        self.idx = idx
        return self.W[idx]

    def backward(self, dout):
        gW = np.zeros_like(self.W)
        np.add.at(gW, self.idx, dout)
        self.grads["W"] = gW
        return None  # indices have no gradient


class LayerNorm:
    """Normalise the last axis, then scale/shift: y = gamma*xhat + beta."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.grads = {}

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def forward(self, x):
        self.mu = x.mean(-1, keepdims=True)
        self.xc = x - self.mu
        self.var = (self.xc**2).mean(-1, keepdims=True)
        self.istd = 1.0 / np.sqrt(self.var + self.eps)
        self.xhat = self.xc * self.istd
        return self.gamma * self.xhat + self.beta

    def backward(self, dout):
        D = self.xhat.shape[-1]
        self.grads["gamma"] = (dout * self.xhat).reshape(-1, D).sum(0)
        self.grads["beta"] = dout.reshape(-1, D).sum(0)
        dxhat = dout * self.gamma
        # standard LayerNorm input gradient
        dx = self.istd / D * (
            D * dxhat
            - dxhat.sum(-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).sum(-1, keepdims=True)
        )
        return dx


class CausalSelfAttention:
    """Multi-head causal self-attention. dim must be divisible by n_head."""

    def __init__(self, dim, n_head, rng):
        assert dim % n_head == 0
        self.n_head = n_head
        self.hd = dim // n_head
        self.c_attn = Linear(dim, 3 * dim, rng, scale=0.02)  # Q, K, V in one matmul
        self.c_proj = Linear(dim, dim, rng, scale=0.02)
        self.grads = {}

    def params(self):
        return {**{f"c_attn.{k}": v for k, v in self.c_attn.params().items()},
                **{f"c_proj.{k}": v for k, v in self.c_proj.params().items()}}

    def _split(self, t, B, T):
        # (B,T,dim) -> (B, n_head, T, hd)
        return t.reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)

    def forward(self, x):
        B, T, dim = x.shape
        self.B, self.T = B, T
        qkv = self.c_attn.forward(x)                       # (B,T,3dim)
        q, k, v = np.split(qkv, 3, axis=-1)
        self.q, self.k, self.v = (self._split(t, B, T) for t in (q, k, v))
        scale = 1.0 / np.sqrt(self.hd)
        att = (self.q @ self.k.transpose(0, 1, 3, 2)) * scale   # (B,nh,T,T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        att = np.where(mask, -1e9, att)
        self.p = softmax(att, axis=-1)
        self.scale = scale
        y = self.p @ self.v                                # (B,nh,T,hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, dim)     # merge heads
        return self.c_proj.forward(y)

    def backward(self, dout):
        B, T, nh, hd = self.B, self.T, self.n_head, self.hd
        dy = self.c_proj.backward(dout)                    # (B,T,dim)
        dy = dy.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)  # (B,nh,T,hd)
        dp = dy @ self.v.transpose(0, 1, 3, 2)             # (B,nh,T,T)
        dv = self.p.transpose(0, 1, 3, 2) @ dy
        # softmax backward, row-wise
        ds = self.p * (dp - (dp * self.p).sum(-1, keepdims=True))
        ds *= self.scale
        dq = ds @ self.k
        dk = ds.transpose(0, 1, 3, 2) @ self.q
        merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        dx = self.c_attn.backward(dqkv)
        self.grads = self.params_grads()
        return dx

    def params_grads(self):
        return {**{f"c_attn.{k}": v for k, v in self.c_attn.grads.items()},
                **{f"c_proj.{k}": v for k, v in self.c_proj.grads.items()}}


class MLP:
    """Position-wise feed-forward: Linear -> GELU -> Linear, 4x hidden."""

    def __init__(self, dim, rng):
        self.fc = Linear(dim, 4 * dim, rng, scale=0.02)
        self.proj = Linear(4 * dim, dim, rng, scale=0.02)
        self.grads = {}

    def params(self):
        return {**{f"fc.{k}": v for k, v in self.fc.params().items()},
                **{f"proj.{k}": v for k, v in self.proj.params().items()}}

    def forward(self, x):
        self.h = self.fc.forward(x)
        return self.proj.forward(gelu(self.h))

    def backward(self, dout):
        dg = self.proj.backward(dout)
        dx = self.fc.backward(dg * dgelu(self.h))
        self.grads = {**{f"fc.{k}": v for k, v in self.fc.grads.items()},
                      **{f"proj.{k}": v for k, v in self.proj.grads.items()}}
        return dx


class Block:
    """Pre-norm transformer block with residual connections."""

    def __init__(self, dim, n_head, rng):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, rng)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, rng)
        self.grads = {}

    def params(self):
        return {**{f"ln1.{k}": v for k, v in self.ln1.params().items()},
                **{f"attn.{k}": v for k, v in self.attn.params().items()},
                **{f"ln2.{k}": v for k, v in self.ln2.params().items()},
                **{f"mlp.{k}": v for k, v in self.mlp.params().items()}}

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, dout):
        dmlp = self.ln2.backward(self.mlp.backward(dout))
        dout = dout + dmlp                       # residual around the MLP branch
        dattn = self.ln1.backward(self.attn.backward(dout))
        dx = dout + dattn                        # residual around the attn branch
        self.grads = {
            **{f"ln1.{k}": v for k, v in self.ln1.grads.items()},
            **{f"attn.{k}": v for k, v in self.attn.grads.items()},
            **{f"ln2.{k}": v for k, v in self.ln2.grads.items()},
            **{f"mlp.{k}": v for k, v in self.mlp.grads.items()}}
        return dx


class GPT:
    """The full model. Config: vocab_size, block_size, n_layer, n_head, n_embd."""

    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=128, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.wte = Embedding(vocab_size, n_embd, rng)
        self.wpe = Embedding(block_size, n_embd, rng)
        self.blocks = [Block(n_embd, n_head, rng) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size, rng, scale=0.02)
        self.grads = {}

    def params(self):
        p = {**{f"wte.{k}": v for k, v in self.wte.params().items()},
             **{f"wpe.{k}": v for k, v in self.wpe.params().items()},
             **{f"ln_f.{k}": v for k, v in self.ln_f.params().items()},
             **{f"head.{k}": v for k, v in self.head.params().items()}}
        for i, b in enumerate(self.blocks):
            p.update({f"block{i}.{k}": v for k, v in b.params().items()})
        return p

    def forward(self, idx):
        B, T = idx.shape
        pos = np.arange(T)
        x = self.wte.forward(idx) + self.wpe.forward(pos)   # (B,T,n_embd)
        for b in self.blocks:
            x = b.forward(x)
        x = self.ln_f.forward(x)
        return self.head.forward(x)                         # logits (B,T,vocab)

    def loss(self, idx, targets):
        """Mean cross-entropy over all positions; caches for backward()."""
        logits = self.forward(idx)
        B, T, V = logits.shape
        self.probs = softmax(logits, axis=-1)
        self.targets = targets
        ll = np.log(self.probs.reshape(-1, V)[np.arange(B * T), targets.reshape(-1)] + 1e-12)
        return -ll.mean()

    def backward(self):
        B, T, V = self.probs.shape
        dlogits = self.probs.copy()
        dlogits.reshape(-1, V)[np.arange(B * T), self.targets.reshape(-1)] -= 1.0
        dlogits /= (B * T)                                  # gradient of the mean
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.wte.backward(dx)
        self.wpe.backward(dx.sum(0))                        # pos emb shared over batch

        g = {**{f"wte.{k}": v for k, v in self.wte.grads.items()},
             **{f"wpe.{k}": v for k, v in self.wpe.grads.items()},
             **{f"ln_f.{k}": v for k, v in self.ln_f.grads.items()},
             **{f"head.{k}": v for k, v in self.head.grads.items()}}
        for i, b in enumerate(self.blocks):
            g.update({f"block{i}.{k}": v for k, v in b.grads.items()})
        self.grads = g
        return g
