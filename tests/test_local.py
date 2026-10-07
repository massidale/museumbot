import math

import mlx.core as mx
import numpy as np
import pytest

from museumbot.generation.local import (
    check_pilot,
    decode,
    log_softmax,
    mix_logprobs,
    row_seed,
)

V = 6  # vocabolario del modello finto
EOS = 5


class FakeCache:
    def __init__(self):
        self.tokens = []


class FakeModel:
    """Logit deterministici: dipendono dal primo token del prompt (l'"esperto") e
    dall'ultimo token visto. Il primo token 0 preferisce salire, 1 preferisce scendere."""

    def __call__(self, inputs, cache):
        cache.tokens += inputs[0].tolist()
        first, last = cache.tokens[0], cache.tokens[-1]
        up = 1 if first == 0 else -1
        target = (last + up) % (V - 1)
        logits = np.full(V, -4.0, dtype=np.float32)
        logits[target] = 4.0
        if len(cache.tokens) >= 8:
            logits[EOS] = 10.0
        return mx.array(logits)[None, None, :]


def run(prompts, weights, seed=0, max_tokens=20):
    return decode(FakeModel(), prompts, weights, temp=0.7, seed=seed,
                  max_tokens=max_tokens, eos={EOS}, make_cache=FakeCache)


def test_mix_logprobs_one_hot_returns_that_expert():
    a = log_softmax(mx.array([1.0, 2.0, 3.0]))
    b = log_softmax(mx.array([3.0, 0.0, 0.0]))
    out = mix_logprobs([a, b], [1.0, 0.0])
    assert np.allclose(np.array(out), np.array(a), atol=1e-6)


def test_mix_logprobs_is_normalized_geometric_mean():
    a = log_softmax(mx.array([1.0, 2.0, 3.0]))
    b = log_softmax(mx.array([3.0, 0.0, 0.0]))
    out = np.array(mix_logprobs([a, b], [0.5, 0.5]))
    assert math.isclose(float(np.exp(out).sum()), 1.0, rel_tol=1e-5)
    want = 0.5 * np.array(a) + 0.5 * np.array(b)
    want -= np.log(np.exp(want).sum())
    assert np.allclose(out, want, atol=1e-5)


@pytest.mark.parametrize("w", [[0.5, 0.6], [1.2, -0.2], [1.0]])
def test_mix_logprobs_rejects_bad_weights(w):
    lp = [log_softmax(mx.zeros(3)), log_softmax(mx.zeros(3))]
    with pytest.raises(ValueError):
        mix_logprobs(lp, w)


def test_decode_one_hot_equals_single_expert():
    both, _ = run([[0], [1]], [1.0, 0.0])
    alone, _ = run([[0]], [1.0])
    assert both == alone


def test_decode_follows_the_heavier_expert():
    up, _ = run([[0], [1]], [1.0, 0.0])
    down, _ = run([[0], [1]], [0.0, 1.0])
    assert up[:3] == [1, 2, 3]
    assert down[:3] == [0, 4, 3]  # l'esperto 1 parte dal suo prompt [1] e scende


def test_decode_is_reproducible_with_seed():
    assert run([[0], [1]], [0.5, 0.5], seed=3) == run([[0], [1]], [0.5, 0.5], seed=3)


def test_decode_stops_at_eos_and_max_tokens():
    toks, diag = run([[0]], [1.0])
    assert EOS not in toks and len(toks) == 7  # prompt di 1 token + 7 generati = 8
    toks, diag = run([[0]], [1.0], max_tokens=3)
    assert len(toks) == 3


def test_decode_diag_has_one_entry_per_token_and_expert():
    toks, diag = run([[0], [1]], [0.5, 0.5], seed=1)
    assert len(diag) == len(toks)
    assert all(len(d["logp"]) == 2 and d["div"] >= 0 for d in diag)
    one, diag1 = run([[0]], [1.0])
    assert all(d["div"] == pytest.approx(0.0, abs=1e-6) for d in diag1)


def test_row_seed_stable_and_distinct():
    assert row_seed("Q1", "explorer", "full") == row_seed("Q1", "explorer", "full")
    assert row_seed("Q1", "explorer", "full") != row_seed("Q1", "explorer", "full_rep")


def row(text, words=None, raw=None, attempts=1):
    return {"text": text, "words": words or len(text.split()),
            "usage": {"raw": raw or text, "attempts": attempts}}


def test_check_pilot_passes_clean_rows():
    rows = [row("word " * 240 + "end.") for _ in range(30)]
    rep = check_pilot(rows)
    assert rep["passed"] and rep["truncated"] == 0 and rep["violations"] == 0


def test_check_pilot_counts_markdown_retries_and_truncation():
    ok = "word " * 240 + "end."
    rows = [row(ok) for _ in range(27)]
    rows += [row(ok, raw="**Title**\n" + ok), row(ok, attempts=2), row("word " * 240 + "cut")]
    rep = check_pilot(rows)
    assert rep["violations"] == 2 and rep["truncated"] == 1
    assert rep["passed"]
    rows.append(row("word " * 240 + "cut"))
    assert not check_pilot(rows)["passed"]


def test_check_pilot_fails_on_short_texts():
    rows = [row("word " * 240 + "end.") for _ in range(29)] + [row("word " * 100 + "end.")]
    assert not check_pilot(rows)["passed"]
