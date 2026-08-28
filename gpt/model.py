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


def log_softmax(x, axis=-1):
    """log(softmax(x)) computed stably as z - log(sum(exp(z))), z = x - max."""
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


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


class Dropout:
    """Inverted dropout. Training: zero each element with probability p and
    scale the survivors by 1/(1-p) so the expected activation is unchanged;
    the backward pass is the same mask (d(x*m)/dx = m). Eval, or p=0: identity.
    Masks come from `rng`, which the model can reseed for gradient checking."""

    def __init__(self, p, rng):
        self.p, self.rng, self.training = p, rng, True
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


class CausalSelfAttention:
    """Multi-head causal self-attention. dim must be divisible by n_head."""

    def __init__(self, dim, n_head, rng, dropout=0.0, drop_rng=None):
        assert dim % n_head == 0
        self.n_head = n_head
        self.hd = dim // n_head
        self.c_attn = Linear(dim, 3 * dim, rng, scale=0.02)  # Q, K, V in one matmul
        self.c_proj = Linear(dim, dim, rng, scale=0.02)
        self.attn_drop = Dropout(dropout, drop_rng)          # on the attention weights
        self.grads = {}

    def params(self):
        return {**{f"c_attn.{k}": v for k, v in self.c_attn.params().items()},
                **{f"c_proj.{k}": v for k, v in self.c_proj.params().items()}}

    def _split(self, t, B, T):
        # (B,T,dim) -> (B, n_head, T, hd)
        return t.reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)

    def forward(self, x, cache=None):
        """x: (B,T,dim). With `cache` (a dict) the keys/values of earlier
        tokens are read from and appended to it, so x may hold only the *new*
        tokens — this is what makes autoregressive sampling O(T) per step
        instead of O(T^2). The training path (cache=None) is unchanged."""
        B, T, dim = x.shape
        self.B, self.T = B, T
        qkv = self.c_attn.forward(x)                       # (B,T,3dim)
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = (self._split(t, B, T) for t in (q, k, v))
        if cache is not None:
            if "k" in cache:
                k = np.concatenate([cache["k"], k], axis=2)
                v = np.concatenate([cache["v"], v], axis=2)
            cache["k"], cache["v"] = k, v
        self.q, self.k, self.v = q, k, v
        Tk = k.shape[2]                                    # keys = past + new
        past = Tk - T
        scale = 1.0 / np.sqrt(self.hd)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale        # (B,nh,T,Tk)
        # query i sits at absolute position past+i and may see keys <= past+i
        mask = np.triu(np.ones((T, Tk), dtype=bool), k=past + 1)
        att = np.where(mask, -1e9, att)
        self.p = softmax(att, axis=-1)
        self.scale = scale
        self.pd = self.attn_drop.forward(self.p)           # dropped-out weights
        y = self.pd @ v                                    # (B,nh,T,hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, dim)     # merge heads
        return self.c_proj.forward(y)

    def backward(self, dout):
        B, T, nh, hd = self.B, self.T, self.n_head, self.hd
        dy = self.c_proj.backward(dout)                    # (B,T,dim)
        dy = dy.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)  # (B,nh,T,hd)
        dpd = dy @ self.v.transpose(0, 1, 3, 2)            # (B,nh,T,T) w.r.t. dropped weights
        dv = self.pd.transpose(0, 1, 3, 2) @ dy
        dp = self.attn_drop.backward(dpd)                  # through the dropout mask
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
    """Pre-norm transformer block with residual connections. Dropout sits on
    each branch's output before it is added back (GPT-2's resid_dropout)."""

    def __init__(self, dim, n_head, rng, dropout=0.0, drop_rng=None):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, rng, dropout, drop_rng)
        self.drop1 = Dropout(dropout, drop_rng)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, rng)
        self.drop2 = Dropout(dropout, drop_rng)
        self.grads = {}

    def params(self):
        return {**{f"ln1.{k}": v for k, v in self.ln1.params().items()},
                **{f"attn.{k}": v for k, v in self.attn.params().items()},
                **{f"ln2.{k}": v for k, v in self.ln2.params().items()},
                **{f"mlp.{k}": v for k, v in self.mlp.params().items()}}

    def forward(self, x, cache=None):
        x = x + self.drop1.forward(self.attn.forward(self.ln1.forward(x), cache))
        x = x + self.drop2.forward(self.mlp.forward(self.ln2.forward(x)))
        return x

    def backward(self, dout):
        dmlp = self.ln2.backward(self.mlp.backward(self.drop2.backward(dout)))
        dout = dout + dmlp                       # residual around the MLP branch
        dattn = self.ln1.backward(self.attn.backward(self.drop1.backward(dout)))
        dx = dout + dattn                        # residual around the attn branch
        self.grads = {
            **{f"ln1.{k}": v for k, v in self.ln1.grads.items()},
            **{f"attn.{k}": v for k, v in self.attn.grads.items()},
            **{f"ln2.{k}": v for k, v in self.ln2.grads.items()},
            **{f"mlp.{k}": v for k, v in self.mlp.grads.items()}}
        return dx


class GPT:
    """The full model. Config: vocab_size, block_size, n_layer, n_head, n_embd.
    `dropout` is a training-time regulariser (0 = off); call eval() before
    scoring or sampling and train() to switch it back on."""

    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=128, seed=0,
                 dropout=0.0):
        rng = np.random.default_rng(seed)
        self.drop_rng = np.random.default_rng(seed + 1)     # masks; reseedable
        self.block_size = block_size
        self.wte = Embedding(vocab_size, n_embd, rng)
        self.wpe = Embedding(block_size, n_embd, rng)
        self.drop = Dropout(dropout, self.drop_rng)         # on the summed embeddings
        self.blocks = [Block(n_embd, n_head, rng, dropout, self.drop_rng)
                       for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size, rng, scale=0.02)
        self.grads = {}

    # -- dropout control
    def dropouts(self):
        ds = [self.drop]
        for b in self.blocks:
            ds += [b.attn.attn_drop, b.drop1, b.drop2]
        return ds

    def train(self):
        for d in self.dropouts():
            d.training = True
        return self

    def eval(self):
        for d in self.dropouts():
            d.training = False
        return self

    def set_dropout(self, p):
        for d in self.dropouts():
            d.p = p
        return self

    def reseed_dropout(self, seed):
        """Restart the mask stream, so two consecutive forward passes draw
        identical masks — what the gradient check needs."""
        self.drop_rng = np.random.default_rng(seed)
        for d in self.dropouts():
            d.rng = self.drop_rng
        return self

    def params(self):
        p = {**{f"wte.{k}": v for k, v in self.wte.params().items()},
             **{f"wpe.{k}": v for k, v in self.wpe.params().items()},
             **{f"ln_f.{k}": v for k, v in self.ln_f.params().items()},
             **{f"head.{k}": v for k, v in self.head.params().items()}}
        for i, b in enumerate(self.blocks):
            p.update({f"block{i}.{k}": v for k, v in b.params().items()})
        return p

    def new_cache(self):
        """One KV cache per block, for forward(idx, cache=...)."""
        return [{} for _ in self.blocks]

    @staticmethod
    def cache_len(cache):
        return cache[0]["k"].shape[2] if cache and "k" in cache[0] else 0

    def forward(self, idx, cache=None):
        """idx: (B,T) token ids -> logits (B,T,vocab). With a cache from
        new_cache(), idx holds only tokens not yet in the cache and positions
        continue from where the cache left off."""
        B, T = idx.shape
        past = self.cache_len(cache)
        if past + T > self.block_size:
            raise ValueError(f"sequence length {past + T} exceeds block_size {self.block_size}; "
                             "crop the input to the last block_size tokens")
        pos = np.arange(past, past + T)
        x = self.drop.forward(self.wte.forward(idx) + self.wpe.forward(pos))  # (B,T,n_embd)
        for i, b in enumerate(self.blocks):
            x = b.forward(x, cache[i] if cache is not None else None)
        x = self.ln_f.forward(x)
        return self.head.forward(x)                         # logits (B,T,vocab)

    def loss(self, idx, targets):
        """Mean cross-entropy over all positions; caches for backward().

        Computed as a log-softmax (logits - logsumexp) rather than log(softmax),
        so the result is exact even when the softmax underflows to 0 for the
        target class — no epsilon fudge needed, and the gradient in backward()
        is still just probs - onehot.
        """
        logits = self.forward(idx)
        B, T, V = logits.shape
        self.targets = targets
        logp = log_softmax(logits, axis=-1)
        self.probs = np.exp(logp)
        ll = logp.reshape(-1, V)[np.arange(B * T), targets.reshape(-1)]
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
        dx = self.drop.backward(dx)
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
