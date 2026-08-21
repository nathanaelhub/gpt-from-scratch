"""
Tests for the from-scratch GPT.

The load-bearing test is the gradient check: it proves every hand-derived
backward pass matches finite differences, i.e. the transformer's backprop is
correct. The rest pin the activation derivative, output shapes, causality (no
attending to the future), and that the model actually trains.
"""
import numpy as np
import pytest

from gpt.data import CharData, encode
from gpt.gradcheck import gradcheck
from gpt.model import GPT, dgelu, gelu, log_softmax, softmax
from gpt.optim import Adam, clip_grad_norm, lr_at


def test_gradients_match_finite_differences():
    worst = gradcheck(verbose=False)
    assert worst < 1e-4, f"gradient check failed: worst relative error {worst:.2e}"


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(0).standard_normal((4, 7))
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(-1), 1.0)
    assert (p >= 0).all()


def test_softmax_is_shift_invariant_and_stable():
    x = np.array([[1000.0, 1001.0, 1002.0]])   # would overflow without the max trick
    p = softmax(x)
    assert np.isfinite(p).all() and np.allclose(p.sum(), 1.0)


def test_log_softmax_matches_log_of_softmax_and_is_exact_at_extremes():
    x = np.random.default_rng(1).standard_normal((3, 9))
    assert np.allclose(log_softmax(x), np.log(softmax(x)))
    # a target class 1e3 below the max: softmax underflows to exactly 0, but the
    # log-probability is still finite and correct (-1000 - log(1 + e^-1000)).
    x = np.array([[0.0, -1000.0]])
    assert softmax(x)[0, 1] == 0.0
    assert np.isfinite(log_softmax(x)).all()
    assert np.isclose(log_softmax(x)[0, 1], -1000.0)


def test_loss_is_finite_with_extreme_logits():
    m = GPT(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8, seed=0)
    m.head.W *= 1e4        # blow the logits up so most probabilities underflow
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5, (2, 4))
    targets = rng.integers(0, 5, (2, 4))
    loss = m.loss(idx, targets)
    assert np.isfinite(loss)
    grads = m.backward()
    assert all(np.isfinite(g).all() for g in grads.values())


def test_gelu_derivative_matches_numeric():
    x = np.linspace(-3, 3, 25)
    eps = 1e-6
    num = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    assert np.allclose(dgelu(x), num, atol=1e-6)


def test_forward_shapes():
    m = GPT(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16)
    idx = np.zeros((3, 8), dtype=np.int64)
    logits = m.forward(idx)
    assert logits.shape == (3, 8, 20)


def test_forward_rejects_sequences_longer_than_block_size():
    m = GPT(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    with pytest.raises(ValueError, match="block_size"):
        m.forward(np.zeros((1, 9), dtype=np.int64))


def test_encode_rejects_characters_outside_the_vocab():
    stoi = {c: i for i, c in enumerate("abc ")}
    assert encode("cab", stoi).tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="'z'"):
        encode("a z", stoi)


def test_attention_is_causal():
    # changing the LAST token must not alter the logits at earlier positions
    m = GPT(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16, seed=1)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 20, (1, 8))
    a = m.forward(idx).copy()
    idx2 = idx.copy()
    idx2[0, -1] = (idx2[0, -1] + 1) % 20
    b = m.forward(idx2)
    assert np.allclose(a[:, :-1], b[:, :-1])   # earlier positions unchanged
    assert not np.allclose(a[:, -1], b[:, -1]) # the last position did change


def test_training_reduces_loss():
    m = GPT(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16, seed=0)
    opt = Adam(m.params(), lr=3e-3)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, (8, 8))
    y = rng.integers(0, 20, (8, 8))
    first = m.loss(x, y)
    for _ in range(40):
        m.loss(x, y)
        opt.step(m.backward())
    assert m.loss(x, y) < first - 0.5   # clearly learned the fixed batch


def test_char_data_roundtrip(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("hello world\nhello again\n")
    d = CharData(str(p), block_size=4)
    assert d.decode(d.encode("hello")) == "hello"
    x, y = d.get_batch("train", 3)
    assert x.shape == (3, 4) and y.shape == (3, 4)
    assert (y[:, :-1] == x[:, 1:]).all()   # targets are inputs shifted by one


def test_clip_grad_norm_bounds_the_global_norm_and_keeps_direction():
    rng = np.random.default_rng(0)
    grads = {"a": rng.standard_normal((5, 5)) * 10, "b": rng.standard_normal(5) * 10}
    before = {k: v.copy() for k, v in grads.items()}
    pre = clip_grad_norm(grads, 1.0)
    post = np.sqrt(sum((g * g).sum() for g in grads.values()))
    assert pre > 1.0 and np.isclose(post, 1.0, atol=1e-5)
    for k in grads:                       # same direction, just rescaled
        assert np.allclose(grads[k] / post, before[k] / pre)
    # already small: untouched
    small = {"a": np.full(3, 0.1)}
    clip_grad_norm(small, 1.0)
    assert np.allclose(small["a"], 0.1)


def test_lr_schedule_warms_up_then_cosine_decays_to_min():
    peak, total, warm, floor = 1e-2, 1000, 100, 1e-3
    sched = [lr_at(s, peak, total, warm, floor) for s in range(1, total + 1)]
    assert np.isclose(sched[0], peak / warm)          # step 1 of warmup
    assert np.isclose(sched[warm - 1], peak)          # reaches the peak
    assert np.isclose(sched[-1], floor)               # ends at min_lr
    assert all(b <= a + 1e-12 for a, b in zip(sched[warm - 1:], sched[warm:]))  # monotone after warmup
    assert all(floor - 1e-12 <= v <= peak + 1e-12 for v in sched[warm:])  # bounded once warm
    # warmup=0 means no ramp: starts at the peak
    assert np.isclose(lr_at(1, peak, total, 0, floor), peak, rtol=1e-4)


def test_weight_decay_is_decoupled_and_skips_1d_params():
    W = np.full((4, 4), 1.0); gamma = np.full(4, 1.0); b = np.full(4, 1.0)
    params = {"W": W, "ln.gamma": gamma, "b": b}
    zero = {k: np.zeros_like(v) for k, v in params.items()}
    opt = Adam(params, lr=0.1, weight_decay=0.5)
    opt.step(zero)                       # no gradient signal: only decay can move anything
    assert np.allclose(W, 1.0 - 0.1 * 0.5)       # p *= 1 - lr*wd, independent of Adam's v
    assert np.allclose(gamma, 1.0) and np.allclose(b, 1.0)   # 1-D params untouched
    # and with wd=0 nothing moves at all
    W2 = np.full((2, 2), 1.0)
    Adam({"W": W2}, lr=0.1, weight_decay=0.0).step({"W": np.zeros((2, 2))})
    assert np.allclose(W2, 1.0)
