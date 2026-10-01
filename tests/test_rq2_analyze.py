import pytest

from museumbot.rq2_judge.analyze import bootstrap_mean, criteria, paired_diff, pair_scores


def j(pid, typ, persona, order, chose, choice="A", valid=True, a="Q1", target=None):
    target = target or [a, persona]
    return {"pair_id": pid, "artwork_id": a, "type": typ, "persona": persona, "order": order,
            "chose_target": chose if valid else None, "choice": choice if valid else None,
            "valid": valid, "target": target, "other": [a, "flat"]}


def test_pair_score_averages_the_two_orders():
    rows = [j("p1", "A", "explorer", "target_first", True),
            j("p1", "A", "explorer", "target_second", False, "A"),
            j("p2", "A", "recharger", "target_first", True),
            j("p2", "A", "recharger", "target_second", True, "B"),
            j("p3", "A", "facilitator", "target_first", True),
            j("p3", "A", "facilitator", "target_second", None, valid=False)]
    s = pair_scores(rows)
    assert s["p1"]["score"] == 0.5 and not s["p1"]["consistent"]
    assert s["p2"]["score"] == 1.0 and s["p2"]["consistent"]
    assert "p3" not in s  # un ordine non valido: coppia esclusa


def test_criteria():
    rows = [j("a1", "attention", "neutral", "target_first", True, target=["Q1", "flat"]),
            j("a1", "attention", "neutral", "target_second", False, "A", target=["Q1", "flat"]),
            j("p1", "A", "explorer", "target_first", True),
            j("p1", "A", "explorer", "target_second", True, "B")]
    c = criteria(rows)
    assert c["valid_rate"] == 1.0
    assert c["attention_accuracy"] == 0.5
    assert c["position_A_rate"] == 0.75


def test_paired_diff_matches_on_artwork_and_category():
    scores = {
        "Q1|explorer|A": {"artwork_id": "Q1", "type": "A", "persona": "explorer",
                          "target": ["Q1", "explorer"], "score": 1.0},
        "Q1|explorer|C": {"artwork_id": "Q1", "type": "C", "persona": "explorer",
                          "target": ["Q1", "recharger"], "score": 0.5},
        "Q1|explorer|N": {"artwork_id": "Q1", "type": "N", "persona": "neutral",
                          "target": ["Q1", "explorer"], "score": 0.0},
    }
    assert paired_diff(scores, "A", "C") == {"Q1": [0.5]}
    assert paired_diff(scores, "A", "N") == {"Q1": [1.0]}


def test_bootstrap_mean_of_constant_is_exact():
    m, lo, hi = bootstrap_mean({"Q1": [1.0, 1.0], "Q2": [1.0]}, n=50)
    assert m == lo == hi == pytest.approx(1.0)


def test_preference_matrix_uses_a_on_diagonal_and_c_off():
    import numpy as np
    from museumbot.rq2_judge.analyze import preference_matrix

    scores = {
        "a": {"artwork_id": "Q1", "type": "A", "persona": "explorer", "target": ["Q1", "explorer"], "score": 1.0},
        "c": {"artwork_id": "Q1", "type": "C", "persona": "explorer", "target": ["Q1", "recharger"], "score": 0.5},
        "b": {"artwork_id": "Q1", "type": "B", "persona": "explorer", "target": ["Q1", "explorer"], "score": 0.0},
    }
    M, n = preference_matrix(scores)
    assert M[0, 0] == 1.0 and M[0, 4] == 0.5 and n[0, 4] == 1
    assert np.isnan(M[1, 1])
