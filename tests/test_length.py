import numpy as np
import pandas as pd
import pytest

from museumbot.common.prompts import CONDITIONS
from museumbot.rq1_embeddings.analyze import FLAT, steering
from museumbot.rq1_embeddings.length import length_matrix, length_probe, residualize

C = len(CONDITIONS)


def test_length_matrix_follows_artworks_and_conditions():
    meta = pd.DataFrame([{"artwork_id": a, "condition": c, "words": 100 + i}
                         for i, (a, c) in enumerate((a, c) for a in ("Q2", "Q1") for c in CONDITIONS)])
    L = length_matrix(meta, ["Q1", "Q2"])
    assert L.shape == (2, C)
    assert L[1, 0] == pytest.approx(np.log(100))  # Q2, prima condizione
    assert L[0, FLAT] == pytest.approx(np.log(100 + C + FLAT))


def test_residualize_removes_linear_length_effect():
    rng = np.random.default_rng(0)
    A, D = 200, 8
    shift = rng.normal(size=(C, D))
    b = rng.normal(size=D) * 3
    L = rng.normal(size=(A, C))  # lunghezza indipendente dalla condizione
    X = rng.normal(size=(A, 1, D)) * 5 + shift[None] + L[..., None] * b + rng.normal(size=(A, C, D)) * 0.05
    Xr, beta = residualize(X, L)
    assert beta == pytest.approx(b, abs=0.05)
    true = shift - shift[FLAT]
    true = np.delete(true, FLAT, axis=0)
    assert steering(Xr) == pytest.approx(true, abs=0.05)


def test_residualize_applies_given_coefficients():
    X = np.zeros((3, C, 2))
    L = np.tile(np.arange(C, dtype=float), (3, 1))
    Xr, beta = residualize(X, L, beta=np.array([1.0, 0.0]))
    assert beta.tolist() == [1.0, 0.0]
    assert Xr[0, :, 0] == pytest.approx(-(np.arange(C) - np.arange(C).mean()))


def test_length_probe_perfect_when_each_condition_has_its_own_length():
    rng = np.random.default_rng(1)
    L = np.tile(np.arange(C, dtype=float) * 10, (40, 1)) + rng.normal(size=(40, C)) * 0.1
    assert length_probe(L) == pytest.approx(1.0)
