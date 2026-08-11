# GPT from Scratch

[![tests](https://github.com/nathanaelhub/gpt-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/nathanaelhub/gpt-from-scratch/actions/workflows/ci.yml)

A decoder-only transformer — a small GPT — implemented in **NumPy only**. No
PyTorch, no autograd: every gradient, including the ones through multi-head
self-attention and LayerNorm, is derived and coded by hand, then **verified
against numerical finite differences** before any training happens.

It trains on character-level text (tiny-shakespeare by default) and generates
new text one character at a time.

The point is to make `loss.backward()` completely un-magical for a real
transformer — the same spirit as [backprop-from-scratch](https://github.com/nathanaelhub/backprop-from-scratch),
one architecture up.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m gpt.gradcheck          # verify every gradient (~1e-6 error) — do this first
python train.py --steps 2500     # train on tiny-shakespeare (CPU, a few minutes)
python sample.py --prompt "ROMEO:" --n 400
```

## Architecture

Standard GPT-2-style decoder block, pre-norm with residuals:

```mermaid
graph LR
    idx["idx (B,T)"] --> emb["token emb + positional emb"]
    emb --> b1["Block × N"]
    b1 --> lnf["LayerNorm"]
    lnf --> head["Linear → logits (B,T,V)"]
```

Each **Block** is

```
x = x + CausalSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Every layer (`Linear`, `Embedding`, `LayerNorm`, `CausalSelfAttention`, `MLP`,
`Block`, `GPT`) is an object with a `forward` that caches what it needs and a
`backward` that returns the gradient w.r.t. its input and stashes its parameter
gradients — the chain rule, spelled out. See [`gpt/model.py`](gpt/model.py).

## The math that's easy to get wrong

**Causal self-attention.** With per-head queries/keys/values `Q, K, V` of head
dim `d`:

```
S = QKᵀ / √d,  masked so position t sees only ≤ t
P = softmax(S)           A = P V
```

The backward pass threads through the softmax row-wise
(`dS = P ⊙ (dP − (dP·P)Σ)`), the `1/√d` scale, and the head split/merge — all in
[`CausalSelfAttention.backward`](gpt/model.py).

**LayerNorm.** For `ŷ = γ (x − μ)/√(σ²+ε) + β`, the input gradient is the usual

```
dx = (1/√(σ²+ε)) · (dŷ − mean(dŷ) − ŷ · mean(dŷ · ŷ)) / D
```

with `dŷ = dout ⊙ γ`. Getting the two mean-subtraction terms right is the whole
game; the gradient check is what tells you they are.

**Softmax + cross-entropy** collapse, as always, to `dlogits = P − onehot(y)`,
averaged over all `B·T` positions.

## Gradient checking

Before trusting a single training step, `gpt/gradcheck.py` compares each
parameter's analytic gradient to the central difference
`(L(w+ε) − L(w−ε)) / 2ε`. Every parameter agrees to a **max relative error of
~3e-6** — the strongest evidence the calculus is right, independent of whether
training happens to work:

```
$ python -m gpt.gradcheck
  ...
  block0.attn.c_attn.W   max rel err 2.83e-06  [ok]
  block0.ln1.gamma       max rel err 4.60e-08  [ok]
  ...
worst relative error across all parameters: 2.91e-06
PASS
```

## Results

Training the default ~0.6 M-parameter model on tiny-shakespeare (1,200 steps,
a few minutes on CPU), cross-entropy drops from the `ln(vocab) ≈ 4.17` random
baseline to ~1.9, and the samples go from noise to Shakespeare-shaped text —
speaker headings, the play's blank-line structure, and mostly-real words:

```
ROMEO:
That wither the the all shyse shall my'sing,
Which fre coul me of dur surs lows,
I then bows to's us erveres that
ans the will fall the thy nour this.

COPUS:
And you the your russ.

LUCES O:
Now, to in be warwar!

First:
All mare more ervy words, I have lookely.
```

It has clearly learned English word shapes, punctuation, and the speaker/line
format — but it's a tiny character model trained for minutes, so it's not going
to write a real sonnet. More steps and a bigger `--n-embd`/`--n-layer` keep
improving it.

## Project layout

```
gpt-from-scratch/
├── gpt/
│   ├── model.py       # layers + GPT: forward and hand-derived backward
│   ├── optim.py       # Adam
│   ├── data.py        # char-level tokenizer + batching
│   └── gradcheck.py   # numerical gradient verification
├── train.py           # training loop → checkpoint.npz
├── sample.py          # autoregressive generation
├── tests/             # pytest: gradient check, causality, convergence
└── data/              # tiny-shakespeare
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # gradient check + causality + training-reduces-loss + more
```

CI runs the whole suite — including the gradient check — on every push.

## License

MIT — see [LICENSE](LICENSE).
