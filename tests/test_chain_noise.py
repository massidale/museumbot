import numpy as np
import pandas as pd
import pytest

from museumbot.common.prompts import FALK_CATEGORIES
from museumbot.rq1_embeddings.chain_noise import bootstrap, ratio, steering_stats, text_level


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def synthetic(rng, chain_shift=0.0, A=30, D=32, noise=0.3):
    """Embedding per (opera, categoria, metodo): centro di cella + rumore per testo."""
    E, idx = [], {}
    for a in range(A):
        for c in FALK_CATEGORIES:
            center = rng.normal(size=D)
            shift = rng.normal(size=D) * chain_shift
            for m in ("single_a", "single_b", "chain"):
                x = center + rng.normal(size=D) * noise + (shift if m == "chain" else 0)
                idx[(f"Q{a}", c, m)] = len(E)
                E.append(unit(x))
    return np.array(E), idx


def test_ratio_near_one_when_chain_is_just_noise():
    E, idx = synthetic(np.random.default_rng(0))
    df = text_level(E, idx)
    assert ratio(df, "between") == pytest.approx(1.0, abs=0.08)
    lo, hi = bootstrap(df, lambda d: ratio(d, "between"), n=200)
    assert lo < 1.0 < hi


def test_ratio_above_one_when_chain_differs():
    E, idx = synthetic(np.random.default_rng(1), chain_shift=0.4)
    assert ratio(text_level(E, idx), "between") > 1.5


def test_steering_stats_rel_one_for_identical_methods():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 6, 8))
    s = steering_stats(X, X.copy(), X.copy())
    assert np.allclose(s["rel"], 1) and np.allclose(s["norm_ratio"], 1)


def test_ratio_ignores_missing_values():
    df = pd.DataFrame({"within": [0.9, 0.8], "between": [0.85, np.nan]})
    assert ratio(df, "between") == pytest.approx(0.15 / 0.1)
