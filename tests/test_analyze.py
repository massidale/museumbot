import numpy as np
import pytest

from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings import analyze
from museumbot.rq1_embeddings.analyze import FLAT, center, crossval_distance, split_half

C = len(CONDITIONS)


def synthetic(rng, A=40, D=16, noise=0.05, shifts=None):
    """Offset per opera + shift fisso per condizione + rumore per testo."""
    if shifts is None:
        shifts = rng.normal(size=(C, D))
    return rng.normal(size=(A, 1, D)) * 5 + shifts[None] + rng.normal(size=(A, C, D)) * noise


def test_center_mean_zeroes_condition_mean():
    X = synthetic(np.random.default_rng(0))
    assert np.allclose(center(X, "mean").mean(axis=1), 0, atol=1e-9)


def test_center_flat_zeroes_flat_row():
    X = synthetic(np.random.default_rng(0))
    Xf = center(X, "flat")
    assert np.allclose(Xf[:, FLAT], 0)
    assert np.allclose(Xf[:, 0], X[:, 0] - X[:, FLAT])


def test_center_rejects_unknown_ref():
    with pytest.raises(ValueError):
        center(np.zeros((2, C, 3)), "median")


def test_split_half_shapes_and_high_consistency():
    X = synthetic(np.random.default_rng(1))
    mu_f, _ = split_half(X, "flat", n_rep=20)
    mu_m, _ = split_half(X, "mean", n_rep=20)
    assert mu_f.shape == (C - 1,) and mu_m.shape == (C,)
    assert np.all(mu_f > 0.95) and np.all(mu_m > 0.95)


def test_crossval_distance_zero_for_identical_conditions():
    rng = np.random.default_rng(2)
    shifts = rng.normal(size=(C, 16))
    shifts[1] = shifts[0]  # condizioni 0 e 1 indistinguibili
    X = synthetic(rng, noise=0.5, shifts=shifts)
    d2, _ = crossval_distance(X, n_rep=50)
    assert np.allclose(np.diag(d2), 0)
    assert np.allclose(d2, d2.T)
    assert abs(d2[0, 1]) < 0.05
    true = np.sum((shifts[0] - shifts[2]) ** 2)
    assert d2[0, 2] == pytest.approx(true, rel=0.15)


def test_crossval_distance_ignores_reference(monkeypatch):
    X = synthetic(np.random.default_rng(3))
    monkeypatch.setattr(analyze, "RNG", np.random.default_rng(7))
    a, _ = crossval_distance(center(X, "mean"), n_rep=10)
    monkeypatch.setattr(analyze, "RNG", np.random.default_rng(7))
    b, _ = crossval_distance(center(X, "flat"), n_rep=10)
    assert np.allclose(a, b, atol=1e-9)
