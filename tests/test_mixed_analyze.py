import pytest

from museumbot.rq2_judge.mixed_analyze import crossing, rank_spearman, rates, summarize_pair

A = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_crossing_interpolates_where_x_curve_falls_below_y():
    assert crossing(A, [1.0, 0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8, 1.0]) == pytest.approx(0.5)
    assert crossing(A, [1.0, 1.0, 0.9, 0.5, 0.5], [0.5, 0.6, 0.7, 0.7, 1.0]) == pytest.approx(0.625)
    assert crossing(A, [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) is None


def test_rank_spearman_one_for_perfect_and_minus_one_for_reversed():
    assert rank_spearman([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(1.0)
    assert rank_spearman([1.0, 0.75, 0.5, 0.25, 0.0]) == pytest.approx(-1.0)


def row(typ, persona, alpha, chose, pair=("x", "y"), valid=True):
    return {"type": typ, "persona": persona, "alpha": alpha, "chose_target": chose,
            "pair": list(pair), "valid": valid}


def test_rates_by_persona_and_alpha_skip_invalid():
    rows = [row("curve", "x", 0.0, True), row("curve", "x", 0.0, False),
            row("curve", "x", 0.0, True, valid=False), row("curve", "y", 0.0, True)]
    r = rates(rows, "curve")
    assert r[("x", 0.0)] == (0.5, 2) and r[("y", 0.0)] == (1.0, 1)


def test_summarize_pair_curve_double_coherence_rank():
    rows = []
    for al, px, py in zip(A, [1, 1, 1, 0.5, 0.5], [0.5, 0.5, 1, 1, 1]):
        for k in range(4):
            rows.append(row("curve", "x", al, k < 4 * px))
            rows.append(row("curve", "y", al, k < 4 * py))
            rows.append(row("double", "x", al, k < 4 * (1 - al)))
            rows.append(row("double", "y", al, k < 4 * al))
    for al in (0.25, 0.5, 0.75):
        rows += [row("coherence", "neutral", al, k % 2 == 0) for k in range(4)]
    rows += [{"type": "rank", "pair": ["x", "y"], "valid": True,
              "ranked_alphas": [0.0, 0.25, 0.5, 0.75, 1.0]}]
    s = summarize_pair(rows, ("x", "y"))
    assert s["curve"]["spearman_x"] < 0 < s["curve"]["spearman_y"]
    assert s["curve"]["crossing"] == pytest.approx(0.5)
    assert s["double"]["argmax_min"] == 0.5
    assert s["coherence"]["rate"] == pytest.approx(0.5)
    assert s["rank"]["spearman_median"] == pytest.approx(1.0)
