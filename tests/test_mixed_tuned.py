import math

import numpy as np

from museumbot.rq1_embeddings.mixed_tuned import distance, summarize, unlikely_tokens, winners


def test_distance_off_curve_is_farthest():
    assert math.isclose(distance(0.3, 0.25), 0.05)
    assert distance(None, 0.25) == math.inf


def test_unlikely_tokens_needs_both_experts_low():
    diag = [{"logp": [-5.0, -6.0]}, {"logp": [-5.0, -0.1]}, {"logp": [-0.1, -0.2]}]
    assert unlikely_tokens(diag) == 1


def test_summarize_counts_near_and_off_curve():
    curve = {"0": {"alpha_eff_mean": 0.0, "position_median": 0.0},
             "1": {"alpha_eff_mean": 1.0, "position_median": 1.0}}
    s = summarize([0.2, 0.5, None], np.array([0.2, 0.5, 1.2]), [0, 2, 0], 0.25, curve)
    assert s["position_median"] == 0.5 and s["share_of_median"] == 0.5
    assert s["near_target"] == 1 and s["off_curve"] == 1 and s["with_unlikely_token"] == 1


def test_winners_prefers_the_closer_share():
    by = {"0.25": {"target": 0.25, "share_of_median": 0.05},
          "0.4": {"target": 0.25, "share_of_median": 0.3},
          "0.75": {"target": 0.75, "share_of_median": 0.8},
          "0.6": {"target": 0.75, "share_of_median": None}}
    assert winners(by) == {"0.4_vs_0.25": True, "0.6_vs_0.75": False}
