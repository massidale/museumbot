import numpy as np
import pytest

from museumbot.common.prompts import CONDITIONS
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
    sh = split_half(X)
    assert sh.shape == (C - 1,)
    assert np.all(sh > 0.95)


def test_crossval_distance_zero_for_identical_conditions():
    rng = np.random.default_rng(2)
    shifts = rng.normal(size=(C, 16))
    shifts[1] = shifts[0]  # condizioni 0 e 1 indistinguibili
    X = synthetic(rng, noise=0.5, shifts=shifts)
    d2 = crossval_distance(X)
    assert np.allclose(np.diag(d2), 0)
    assert np.allclose(d2, d2.T)
    true = np.sum((shifts[0] - shifts[2]) ** 2)
    assert abs(d2[0, 1]) < 0.02 * true
    assert d2[0, 2] == pytest.approx(true, rel=0.15)


def test_crossval_distance_ignores_reference():
    X = synthetic(np.random.default_rng(3))
    a = crossval_distance(center(X, "mean"))
    b = crossval_distance(center(X, "flat"))
    assert np.allclose(a, b, atol=1e-9)


def test_ceiling_over_replicas_averages_all_pairs():
    from itertools import combinations

    from museumbot.rq1_embeddings.analyze import ceiling_over_replicas, replicate_ceiling

    rng = np.random.default_rng(0)
    runs = [(rng.normal(size=(6, 6, 4)), [f"Q{i}" for i in range(6)]) for _ in range(3)]
    got, n_pairs = ceiling_over_replicas(runs)
    want = np.mean([replicate_ceiling(a[0], a[1], b[0], b[1]) for a, b in combinations(runs, 2)],
                   axis=0)
    assert n_pairs == 3
    assert np.allclose(got, want)
