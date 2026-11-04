import numpy as np

from museumbot.rq1_embeddings.mixed_pick import mean_distance_ranks, pick


def test_mean_distance_ranks_closest_first():
    D = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.05]])
    r = mean_distance_ranks(D)
    assert r[2] == 4  # l'unica lontana dalla media


def test_pick_by_mean_rank_and_smoke():
    arts = ["a", "b", "c"]
    ranks = {"p1": [np.array([1, 2, 3]), np.array([2, 1, 3])],
             "p2": [np.array([3, 1, 2]), np.array([3, 1, 2])]}
    out = pick(ranks, arts, 2)
    assert out["pairs"]["p1"] == ["a", "b"] and out["pairs"]["p2"] == ["b", "c"]
    assert out["smoke"] == "b"
