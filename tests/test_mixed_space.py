import numpy as np
import pytest

from museumbot.rq1_embeddings.mixed_space import fold_probs, subset_center


def synthetic(rng, A=40, K=4, D=10, noise=0.3):
    means = rng.normal(size=(K, D)) * 2
    X = rng.normal(size=(A, 1, D)) * 5 + means[None] + rng.normal(size=(A, K, D)) * noise
    return X, means


def test_subset_center_removes_artwork_mean_and_returns_it():
    X = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    Xc, mu = subset_center(X)
    assert np.allclose(Xc.mean(axis=1), 0) and np.allclose(mu, X.mean(axis=1))


def test_fold_probs_pure_out_of_fold_and_halfway_point_splits_two_classes():
    rng = np.random.default_rng(0)
    X, means = synthetic(rng)
    Xc, mu = subset_center(X)
    arts = np.arange(len(X))
    # testo misto a meta' fra le classi 0 e 1, per ogni opera, centrato come i puri
    M = X[:, 0] * 0.5 + X[:, 1] * 0.5 - mu
    p_pure, p_mix = fold_probs(Xc, M, arts)
    assert p_pure.shape == (len(X), 4, 4) and p_mix.shape == (len(X), 4)
    assert np.mean(p_pure.argmax(axis=2) == np.arange(4)) > 0.95
    avg = p_mix.mean(axis=0)
    assert avg[0] + avg[1] > 0.9
    assert avg[0] == pytest.approx(avg[1], abs=0.2)
