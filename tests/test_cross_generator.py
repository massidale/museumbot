import numpy as np
import pytest

from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings.cross_generator import calibrate, cross_cosines

C = len(CONDITIONS)


def corpus(rng, shifts, A=30, D=12, noise=0.0):
    return rng.normal(size=(A, 1, D)) * 5 + shifts[None] + rng.normal(size=(A, C, D)) * noise


def test_same_shifts_give_cosine_one_on_common_artworks():
    rng = np.random.default_rng(0)
    shifts = rng.normal(size=(C, 12))
    Xa, Xb = corpus(rng, shifts), corpus(rng, shifts)
    arts = [f"Q{i}" for i in range(30)]
    cos, n = cross_cosines(Xa, arts, Xb[::-1], arts[::-1])  # ordine diverso, stesse opere
    assert n == 30 and np.allclose(cos, 1.0)


def test_uses_only_common_artworks():
    rng = np.random.default_rng(1)
    shifts = rng.normal(size=(C, 12))
    Xa, Xb = corpus(rng, shifts), corpus(rng, shifts)
    _, n = cross_cosines(Xa, [f"Q{i}" for i in range(30)], Xb, [f"Q{i}" for i in range(10, 40)])
    assert n == 20


def test_orthogonal_shifts_give_cosine_near_zero():
    rng = np.random.default_rng(2)
    D = 12
    sa = np.zeros((C, D)); sb = np.zeros((C, D))
    for c in range(C - 1):
        sa[c, c] = 1.0
        sb[c, c + 6 if c + 6 < D else 11] = 1.0
    arts = [f"Q{i}" for i in range(30)]
    cos, _ = cross_cosines(corpus(rng, sa), arts, corpus(rng, sb), arts)
    assert np.all(np.abs(cos) < 1e-6)


def test_calibrate_divides_by_geometric_mean_of_ceilings():
    out = calibrate(np.array([0.5, 0.9]), np.array([0.81, 1.0]), np.array([1.0, 0.81]))
    assert out == pytest.approx([0.5 / 0.9, 1.0])
